//! Tests for the conversion of op locations to SPIR-V debug data.
//!
//! Each test builds the same small module. The module has a function, a callee that is inlined
//! twice, a loop, and a branch. The locations have the shape that cubecl gives: a `Named` frame
//! for each function, and a `CallSite` chain for each inlined call.

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{OneRegionInterface, OneResultInterface},
        ops::FuncOp,
        types::{FunctionType, IntegerType, Signedness, UnitType},
    },
    combine::stream::position::SourcePosition,
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    location::{Located, Location, Source},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    value::Value,
};
use pliron_spirv::{
    PlironBuilder,
    ToSpirvOp,
    attrs::VerCapExtAttr,
    ops::{
        BranchConditionalOp,
        BranchOp,
        EntryPointOp,
        ExecutionModeOp,
        FMulOp,
        IAddOp,
        INotEqualOp,
        LoadOp,
        LoopOp,
        MergeOp,
        ReturnOp,
        SelectionOp,
        SpirvModuleOp,
        StoreOp,
        ULessThanOp,
        VariableOp,
    },
    spirv::{AddressingModel, Capability, ExecutionMode, ExecutionModel, MemoryAccess, MemoryModel, StorageClass},
    types::{FloatType, PointerType},
};
use tracel_rspirv::{binary::Assemble, dr::Module};

/// The binary of the test module, emitted by the base commit without debug data.
const BASE_BINARY: &[u8] = include_bytes!("data/debug_info_base.spv");

/// Files and names of the test locations.
struct Frames {
    kernel: Source,
    lib: Source,
}

impl Frames {
    fn new(ctx: &mut Context) -> Self {
        Self {
            kernel: Source::new_from_file(ctx, "src/kernel.rs"),
            lib: Source::new_from_file(ctx, "src/lib.rs"),
        }
    }

    /// A position in the kernel, in the frame of the function `kernel`.
    fn kernel(&self, line: i32, column: i32) -> Location {
        named("kernel", self.kernel, line, column)
    }

    /// A position in `mid`, inlined at a position in the kernel.
    fn mid(&self, kernel_line: i32) -> Location {
        call_site(named("mid", self.lib, 7, 5), self.kernel(kernel_line, 13))
    }

    /// A position in `inner`, inlined in `mid`, inlined at a position in the kernel.
    fn inner(&self, kernel_line: i32) -> Location {
        call_site(named("inner", self.lib, 3, 9), self.mid(kernel_line))
    }
}

fn named(name: &str, src: Source, line: i32, column: i32) -> Location {
    Location::Named {
        name: name.to_string(),
        child_loc: Box::new(Location::SrcPos {
            src,
            pos: SourcePosition { line, column },
        }),
    }
}

fn call_site(callee: Location, at: Location) -> Location {
    Location::CallSite {
        callee: Box::new(callee),
        caller: Box::new(at),
    }
}

/// Appends `op` to `block` with the location `loc`.
fn push<T: Op>(ctx: &mut Context, block: Ptr<BasicBlock>, loc: Location, make: impl FnOnce(&mut Context) -> T) -> T {
    let op = make(ctx);
    let operation = op.get_operation();
    operation.insert_at_back(block, ctx);
    operation.deref_mut(ctx).set_loc(loc);
    op
}

/// Appends a new block with the argument types `args` to the region of `op`.
fn new_block(ctx: &mut Context, op: Ptr<Operation>, args: Vec<TypeHandle>) -> Ptr<BasicBlock> {
    let region = op.deref(ctx).get_region(0);
    let block = BasicBlock::new(ctx, None, args);
    block.insert_at_back(region, ctx);
    block
}

/// Builds the test module for the SPIR-V version `version`.
///
/// ```text
/// fn kernel() {                                   // kernel.rs:1
///     let count = ..; let value = ..;             // kernel.rs:4-5
///     for index in count.. {                      // kernel.rs:6
///         if index != count {                     // kernel.rs:7
///             value = mid(value);                 // kernel.rs:8, mid calls inner
///         }
///         index = index + count;                  // kernel.rs:10
///     }
///     value = mid(value);                         // kernel.rs:12, mid calls inner
///     value = value * value;                      // no location
/// }
/// ```
fn build_module(ctx: &mut Context, version: (u8, u8)) -> SpirvModuleOp {
    let frames = Frames::new(ctx);
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle();
    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless).to_handle();
    let f32_ty = FloatType::get(ctx, 32, None).to_handle();
    let u32_ptr = PointerType::get(ctx, u32_ty, StorageClass::Function).to_handle();
    let f32_ptr = PointerType::get(ctx, f32_ty, StorageClass::Function).to_handle();
    let unit = UnitType::get(ctx).to_handle();
    let func_ty = FunctionType::get(ctx, vec![], vec![unit]);

    let module = SpirvModuleOp::new(ctx, ident("module"), AddressingModel::Logical, MemoryModel::GLSL450);
    module.set_vce(
        ctx,
        VerCapExtAttr {
            version,
            capabilities: vec![Capability::Shader],
            extensions: vec![],
        },
    );
    let module_block = module.get_region(ctx).deref(ctx).get_head().unwrap();

    let func = FuncOp::new(ctx, ident("main"), func_ty);
    push(ctx, module_block, frames.kernel(1, 1), |_| func);
    let entry_point = EntryPointOp::new(
        ctx,
        ExecutionModel::GLCompute,
        ident("main"),
        "main".to_string(),
        vec![],
    );
    push(ctx, module_block, Location::Unknown, |_| entry_point);
    let mode = ExecutionModeOp::new(ctx, ident("main"), ExecutionMode::LocalSize, vec![1, 1, 1]);
    push(ctx, module_block, Location::Unknown, |_| mode);

    let entry = func.get_entry_block(ctx);
    let var_u = push(ctx, entry, frames.kernel(2, 9), |ctx| {
        VariableOp::new(ctx, u32_ptr, StorageClass::Function, None)
    })
    .get_result(ctx);
    let var_f = push(ctx, entry, frames.kernel(3, 9), |ctx| {
        VariableOp::new(ctx, f32_ptr, StorageClass::Function, None)
    })
    .get_result(ctx);
    let count = load(ctx, entry, frames.kernel(4, 13), u32_ty, var_u);
    let value = load(ctx, entry, frames.kernel(5, 13), f32_ty, var_f);

    // The loop: `[entry, header, body, merge]`, as `LoopOp` requires.
    let loop_op = push(ctx, entry, frames.kernel(6, 5), |ctx| LoopOp::new(ctx, vec![]));
    let loop_ptr = loop_op.get_operation();
    let loop_entry = loop_op.entry_block(ctx);
    let header = new_block(ctx, loop_ptr, vec![u32_ty]);
    let body = new_block(ctx, loop_ptr, vec![]);
    let loop_merge = new_block(ctx, loop_ptr, vec![]);
    push(ctx, loop_entry, frames.kernel(6, 5), |ctx| {
        BranchOp::new(ctx, header, vec![count])
    });

    let index = header.deref(ctx).get_argument(0);
    let less = push(ctx, header, frames.kernel(6, 15), |ctx| {
        ULessThanOp::new(ctx, bool_ty, index, count)
    })
    .get_result(ctx);
    let branch = BranchConditionalOp::new(ctx, less, body, vec![], loop_merge, vec![]);
    push(ctx, header, frames.kernel(6, 15), |_| branch);

    // The branch in the loop body: `[entry, then, merge]`.
    let not_equal = push(ctx, body, frames.kernel(7, 12), |ctx| {
        INotEqualOp::new(ctx, bool_ty, index, count)
    })
    .get_result(ctx);
    let selection = push(ctx, body, frames.kernel(7, 9), |ctx| SelectionOp::new(ctx, vec![]));
    let selection_ptr = selection.get_operation();
    let selection_entry = selection.entry_block(ctx);
    let then = new_block(ctx, selection_ptr, vec![]);
    let selection_merge = new_block(ctx, selection_ptr, vec![]);
    let branch = BranchConditionalOp::new(ctx, not_equal, then, vec![], selection_merge, vec![]);
    push(ctx, selection_entry, frames.kernel(7, 9), |_| branch);
    let square = fmul(ctx, then, frames.inner(8), f32_ty, value, value);
    let cube = fmul(ctx, then, frames.mid(8), f32_ty, square, value);
    store(ctx, then, frames.kernel(8, 13), var_f, cube);
    push(ctx, then, frames.kernel(8, 13), |ctx| {
        BranchOp::new(ctx, selection_merge, vec![])
    });
    push(ctx, selection_merge, frames.kernel(9, 9), |ctx| {
        MergeOp::new(ctx, vec![])
    });

    let next = push(ctx, body, frames.kernel(10, 9), |ctx| {
        IAddOp::new(ctx, u32_ty, index, count)
    })
    .get_result(ctx);
    push(ctx, body, frames.kernel(10, 9), |ctx| {
        BranchOp::new(ctx, header, vec![next])
    });
    push(ctx, loop_merge, frames.kernel(11, 5), |ctx| MergeOp::new(ctx, vec![]));

    // The second inlined call of `inner`, and an op without a location.
    let square = fmul(ctx, entry, frames.inner(12), f32_ty, value, value);
    store(ctx, entry, frames.kernel(12, 5), var_f, square);
    let unknown = fmul(ctx, entry, Location::Unknown, f32_ty, square, square);
    store(ctx, entry, Location::Unknown, var_f, unknown);
    push(ctx, entry, frames.kernel(14, 1), |ctx| ReturnOp::new(ctx, None));

    module
}

fn ident(name: &str) -> Identifier {
    name.try_into().unwrap()
}

fn load(ctx: &mut Context, block: Ptr<BasicBlock>, loc: Location, ty: TypeHandle, ptr: Value) -> Value {
    let op = push(ctx, block, loc, |ctx| {
        LoadOp::new(ctx, ty, ptr, MemoryAccess::NONE, None)
    });
    op.get_result(ctx)
}

fn fmul(ctx: &mut Context, block: Ptr<BasicBlock>, loc: Location, ty: TypeHandle, a: Value, b: Value) -> Value {
    let op = push(ctx, block, loc, |ctx| FMulOp::new(ctx, ty, a, b));
    op.get_result(ctx)
}

fn store(ctx: &mut Context, block: Ptr<BasicBlock>, loc: Location, ptr: Value, value: Value) {
    push(ctx, block, loc, |ctx| {
        StoreOp::new(ctx, ptr, value, MemoryAccess::NONE, None)
    });
}

/// Emits the test module for `version` with `builder`.
fn emit(builder: PlironBuilder, version: (u8, u8)) -> Module {
    let ctx = &mut Context::new();
    let module = build_module(ctx, version);
    let mut builder = builder;
    module.to_spirv(ctx, &mut builder).unwrap();
    builder.module()
}

/// Without the option, the output is byte-identical to the output of the base commit.
#[test]
fn no_option_is_byte_identical_to_base() {
    let binary = words_to_bytes(&emit(PlironBuilder::new(), (1, 6)).assemble());
    if std::env::var_os("PLIRON_SPIRV_BLESS").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/debug_info_base.spv");
        std::fs::write(path, &binary).unwrap();
    }
    assert!(binary == BASE_BINARY, "The output without debug data changed");
    tools::validate(&binary, "vulkan1.3");
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// Runs the SPIRV-Tools programs, if they are installed.
///
/// The CI hosts do not have SPIRV-Tools. There, the tests check the module structure only.
mod tools {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };

    /// Runs `tool` with `args` on `binary`. Does nothing if `tool` is not installed.
    fn run(tool: &str, args: &[&str], binary: &[u8]) {
        let Ok(mut child) = Command::new(tool)
            .args(args)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        else {
            eprintln!("{tool} is not installed; skipped");
            return;
        };
        child.stdin.take().unwrap().write_all(binary).unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{tool} failed:\n{stderr}\n{stdout}");
    }

    /// Validates `binary` with `spirv-val` for the target environment `env`.
    pub(crate) fn validate(binary: &[u8], env: &str) {
        run("spirv-val", &["--target-env", env], binary);
    }
}

#[cfg(feature = "debug-info")]
mod with_feature {
    use super::*;
    use pliron::{builtin::ops::FuncOp, location::Located};
    use pliron_spirv::{
        debug_info::{DebugInfoFormat, DebugInfoOptions},
        ops::LineOp,
    };
    use tracel_rspirv::{
        dr::{Instruction, Operand},
        spirv::{DebugInfoOp, Op as SpirvOp, Word},
    };

    fn options(format: DebugInfoFormat) -> DebugInfoOptions {
        let mut options = DebugInfoOptions::default();
        options.format = format;
        options
    }

    /// Emits and validates the test module with `options` for `version` and the target
    /// environment `env`.
    fn emit_valid(options: DebugInfoOptions, version: (u8, u8), env: &str) -> Module {
        let module = emit(PlironBuilder::with_debug_info(options), version);
        let binary = words_to_bytes(&module.assemble());
        tools::validate(&binary, env);
        module
    }

    /// The instructions of the functions of `module`.
    fn function_instructions(module: &Module) -> impl Iterator<Item = &Instruction> {
        module
            .functions
            .iter()
            .flat_map(|func| func.blocks.iter())
            .flat_map(|block| block.label.iter().chain(block.instructions.iter()))
    }

    /// All instructions of `module` that are `op` of `NonSemantic.Shader.DebugInfo.100`.
    fn debug_instructions(module: &Module, op: DebugInfoOp) -> Vec<&Instruction> {
        module
            .types_global_values
            .iter()
            .chain(function_instructions(module))
            .filter(|inst| is_debug(inst, op))
            .collect()
    }

    fn is_debug(inst: &Instruction, op: DebugInfoOp) -> bool {
        inst.class.opcode == SpirvOp::ExtInst && inst.operands[1] == Operand::LiteralExtInstInteger(op as u32)
    }

    /// The `IdRef` operand `index` of the extended instruction `inst`, after the set and the
    /// opcode.
    fn id_operand(inst: &Instruction, index: usize) -> Option<Word> {
        match inst.operands.get(index + 2) {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        }
    }

    fn definition(module: &Module, id: Word) -> &Instruction {
        module
            .types_global_values
            .iter()
            .chain(module.debug_string_source.iter())
            .find(|inst| inst.result_id == Some(id))
            .unwrap_or_else(|| panic!("No definition of %{id}"))
    }

    fn string(module: &Module, id: Word) -> &str {
        match &definition(module, id).operands[0] {
            Operand::LiteralString(text) => text,
            operand => panic!("%{id} is not a string: {operand:?}"),
        }
    }

    fn constant(module: &Module, id: Word) -> u32 {
        match definition(module, id).operands[0] {
            Operand::LiteralBit32(value) => value,
            ref operand => panic!("%{id} is not a constant: {operand:?}"),
        }
    }

    /// The name of the `DebugFunction` `id`.
    fn function_name(module: &Module, id: Word) -> &str {
        string(module, id_operand(definition(module, id), 0).unwrap())
    }

    fn has_extension(module: &Module, name: &str) -> bool {
        module
            .extensions
            .iter()
            .any(|inst| inst.operands == [Operand::LiteralString(name.to_string())])
    }

    /// The frames of a `DebugScope`, innermost first, as function names and lines.
    fn scope_frames(module: &Module, scope: &Instruction) -> Vec<(String, Option<u32>)> {
        let mut frames = vec![(function_name(module, id_operand(scope, 0).unwrap()).to_string(), None)];
        let mut inlined = id_operand(scope, 1);
        while let Some(id) = inlined {
            let inlined_at = definition(module, id);
            assert!(is_debug(inlined_at, DebugInfoOp::DebugInlinedAt));
            let line = constant(module, id_operand(inlined_at, 0).unwrap());
            let name = function_name(module, id_operand(inlined_at, 1).unwrap());
            frames.push((name.to_string(), Some(line)));
            inlined = id_operand(inlined_at, 2);
        }
        frames
    }

    /// `NonSemantic` gives one `DebugFunction` for the function and one for each distinct
    /// callee, and a `DebugInlinedAt` chain for each inlined op.
    #[test]
    fn non_semantic_frames() {
        let module = emit_valid(options(DebugInfoFormat::NonSemantic), (1, 6), "vulkan1.3");

        let mut names = debug_instructions(&module, DebugInfoOp::DebugFunction)
            .into_iter()
            .map(|inst| function_name(&module, inst.result_id.unwrap()))
            .collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(names, ["inner", "kernel", "mid"]);

        let mut scopes = debug_instructions(&module, DebugInfoOp::DebugScope)
            .into_iter()
            .map(|scope| scope_frames(&module, scope))
            .collect::<Vec<_>>();
        scopes.sort();
        scopes.dedup();
        let frame = |name: &str, line: Option<u32>| (name.to_string(), line);
        assert_eq!(
            scopes,
            [
                vec![frame("inner", None), frame("mid", Some(7)), frame("kernel", Some(8))],
                vec![frame("inner", None), frame("mid", Some(7)), frame("kernel", Some(12))],
                vec![frame("kernel", None)],
                vec![frame("mid", None), frame("kernel", Some(8))],
            ]
        );

        assert_eq!(
            debug_instructions(&module, DebugInfoOp::DebugFunctionDefinition).len(),
            1
        );
        assert_eq!(debug_instructions(&module, DebugInfoOp::DebugEntryPoint).len(), 1);
        assert_eq!(debug_instructions(&module, DebugInfoOp::DebugCompilationUnit).len(), 1);
        // The op without a location has no line.
        assert_eq!(debug_instructions(&module, DebugInfoOp::DebugNoLine).len(), 1);
        assert!(!function_instructions(&module).any(|inst| inst.class.opcode == SpirvOp::Line));
        assert!(!has_extension(&module, "SPV_KHR_non_semantic_info"));
    }

    /// Below SPIR-V 1.6, `NonSemantic` adds the extension for non-semantic instruction sets.
    #[test]
    fn non_semantic_spirv_1_3_has_extension() {
        let module = emit_valid(options(DebugInfoFormat::NonSemantic), (1, 3), "vulkan1.1");
        assert!(has_extension(&module, "SPV_KHR_non_semantic_info"));
    }

    /// The `DebugSource` of a file gets the text of the options. A long text continues in
    /// `DebugSourceContinued`. The directory of the options goes before relative file names.
    #[test]
    fn non_semantic_source_text_and_directory() {
        let mut options = options(DebugInfoFormat::NonSemantic);
        // `spirv-val` checks each column against the length of its line in the text.
        let text = format!("// {}\n", "kernel ".repeat(8)).repeat(5_000);
        options.source_text.insert("src/kernel.rs".to_string(), text.clone());
        options.directory = "/work/".to_string();
        let module = emit_valid(options, (1, 6), "vulkan1.3");

        let sources = debug_instructions(&module, DebugInfoOp::DebugSource);
        let mut files = sources
            .iter()
            .map(|inst| string(&module, id_operand(inst, 0).unwrap()))
            .collect::<Vec<_>>();
        files.sort_unstable();
        assert_eq!(files, ["/work/src/kernel.rs", "/work/src/lib.rs"]);

        let kernel = sources
            .iter()
            .find(|inst| string(&module, id_operand(inst, 0).unwrap()) == "/work/src/kernel.rs")
            .unwrap();
        let mut joined = string(&module, id_operand(kernel, 1).unwrap()).to_string();
        let globals = &module.types_global_values;
        let start = globals.iter().position(|inst| inst == *kernel).unwrap();
        for inst in globals[start + 1..]
            .iter()
            .take_while(|inst| is_debug(inst, DebugInfoOp::DebugSourceContinued))
        {
            joined.push_str(string(&module, id_operand(inst, 0).unwrap()));
        }
        assert_eq!(joined, text);
        assert_eq!(debug_instructions(&module, DebugInfoOp::DebugSourceContinued).len(), 1);
    }

    /// `OpLine` gives no `NonSemantic.Shader.DebugInfo.100` instruction.
    #[test]
    fn op_line_has_no_non_semantic_data() {
        let module = emit_valid(options(DebugInfoFormat::OpLine), (1, 6), "vulkan1.3");
        assert!(module.ext_inst_imports.is_empty());
        assert!(function_instructions(&module).any(|inst| inst.class.opcode == SpirvOp::Line));
        // The op without a location has no line.
        assert!(function_instructions(&module).any(|inst| inst.class.opcode == SpirvOp::NoLine));
        emit_valid(options(DebugInfoFormat::OpLine), (1, 3), "vulkan1.1");
    }

    /// The source line of each instruction of the functions, as `OpLine` gives it, in order.
    fn instruction_lines(module: &Module) -> Vec<(SpirvOp, Option<(String, u32)>)> {
        let mut lines = Vec::new();
        for block in module.functions.iter().flat_map(|func| func.blocks.iter()) {
            let mut current = None;
            for inst in &block.instructions {
                match inst.class.opcode {
                    SpirvOp::Line => {
                        let Operand::IdRef(file) = inst.operands[0] else {
                            unreachable!()
                        };
                        let Operand::LiteralBit32(line) = inst.operands[1] else {
                            unreachable!()
                        };
                        current = Some((string(module, file).to_string(), line));
                    }
                    SpirvOp::NoLine => current = None,
                    opcode => lines.push((opcode, current.clone())),
                }
            }
        }
        lines
    }

    /// Gives each op without a location the location of the op before it, as cubecl does
    /// before its `OpLine` pass.
    fn inherit_locations(ctx: &mut Context, block: Ptr<BasicBlock>) {
        let mut previous = Location::Unknown;
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for op in ops {
            if op.deref(ctx).loc().is_unknown() {
                op.deref_mut(ctx).set_loc(previous.clone());
            }
            previous = op.deref(ctx).loc();
            let regions = op.deref(ctx).regions().collect::<Vec<_>>();
            for region in regions {
                let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();
                for block in blocks {
                    inherit_locations(ctx, block);
                }
            }
        }
    }

    /// The innermost file and line of `loc`, as `cubecl_ir::debug::leaf_line` gives it.
    fn leaf_line(ctx: &Context, loc: &Location) -> Option<(String, u32)> {
        match loc {
            Location::CallSite { callee, .. } => leaf_line(ctx, callee),
            Location::Named { child_loc, .. } => leaf_line(ctx, child_loc),
            Location::SrcPos {
                src: Source::File(key),
                pos,
            } => Some((
                pliron::uniqued_any::get(ctx, *key).display().to_string(),
                u32::try_from(pos.line).unwrap_or(0),
            )),
            Location::Fused { locations, .. } => locations.iter().find_map(|loc| leaf_line(ctx, loc)),
            _ => None,
        }
    }

    /// The `OpLine` pass of cubecl (`cubecl-spirv/src/lines.rs` at `0b4f9c9`): a `LineOp` before
    /// each op that is not a terminator and whose line is not the line of the op before it.
    fn insert_line_ops(ctx: &mut Context, block: Ptr<BasicBlock>) {
        let terminator = block.deref(ctx).get_terminator(ctx);
        let mut current = None;
        let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for op in ops {
            let loc = op.deref(ctx).loc();
            if Some(op) != terminator
                && let Some(line) = leaf_line(ctx, &loc)
                && current.as_ref() != Some(&line)
            {
                let (file, number) = line.clone();
                LineOp::new(ctx, file, number, 0u32)
                    .get_operation()
                    .insert_before(ctx, op);
                current = Some(line);
            }
            if op.deref(ctx).num_regions() > 0 {
                let regions = op.deref(ctx).regions().collect::<Vec<_>>();
                for region in regions {
                    let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();
                    for block in blocks {
                        insert_line_ops(ctx, block);
                    }
                }
                current = None;
            }
        }
    }

    /// Emits the test module after cubecl's location pass, with `edit` on the function body.
    fn emit_after_inherit(builder: PlironBuilder, edit: fn(&mut Context, Ptr<BasicBlock>)) -> Module {
        let ctx = &mut Context::new();
        let module = build_module(ctx, (1, 6));
        let module_block = module.get_region(ctx).deref(ctx).get_head().unwrap();
        let func = module_block.deref(ctx).get_head().unwrap();
        let func = Operation::get_op::<FuncOp>(func, ctx).unwrap();
        let blocks = func.get_region(ctx).deref(ctx).iter(ctx).collect::<Vec<_>>();
        for block in blocks {
            inherit_locations(ctx, block);
            edit(ctx, block);
        }
        let mut builder = builder;
        module.to_spirv(ctx, &mut builder).unwrap();
        builder.module()
    }

    /// `OpLine` gives each instruction the same line as the `OpLine` pass of cubecl.
    #[test]
    fn op_line_matches_cubecl() {
        let expected = instruction_lines(&emit_after_inherit(PlironBuilder::new(), insert_line_ops));
        let module = emit_after_inherit(
            PlironBuilder::with_debug_info(options(DebugInfoFormat::OpLine)),
            |_, _| {},
        );
        let actual = instruction_lines(&module);
        assert_eq!(actual, expected);
        assert!(actual.iter().filter(|(_, line)| line.is_some()).count() > 10);
        tools::validate(&words_to_bytes(&module.assemble()), "vulkan1.3");
    }
}
