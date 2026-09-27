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

    /// Runs `tool` with `args` on `binary`, and returns its standard output. `None` if `tool` is
    /// not installed.
    fn run(tool: &str, args: &[&str], binary: &[u8]) -> Option<String> {
        let mut child = match Command::new(tool)
            .args(args)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => {
                eprintln!("{tool} is not installed; skipped");
                return None;
            }
        };
        child.stdin.take().unwrap().write_all(binary).unwrap();
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{tool} failed:\n{stderr}\n{stdout}");
        Some(stdout)
    }

    /// Validates `binary` with `spirv-val` for the target environment `env`.
    pub(crate) fn validate(binary: &[u8], env: &str) {
        run("spirv-val", &["--target-env", env], binary);
    }

    /// The disassembly of `binary` from `spirv-dis`.
    #[allow(dead_code)]
    pub(crate) fn disassemble(binary: &[u8]) -> Option<String> {
        run("spirv-dis", &["--raw-id", "--no-color"], binary)
    }
}

