//! Tests for the textual IR of the dialect: the parser reads what the printer writes.

use pliron::{
    builtin::op_interfaces::SymbolOpInterface,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
};
use pliron_spirv::{PlironBuilder, ToSpirvOp, attrs::VerCapExtAttr, ops::SpirvModuleOp, spirv::Capability};
use tracel_rspirv::binary::Assemble;

/// A module with a function, a loop, a branch, decorations and an entry point.
const MODULE: &str = r#"
spirv.module @module Logical GLSL450 requires : <v1.3, [Shader, Float16], []> {
  ^module_block():
    builtin.func @main: builtin.function <() -> (builtin.unit)> {
      ^entry():
        count_ptr = spirv.Variable Function : <spirv.ptr <builtin.integer ui32, Function>>;
        value_ptr = spirv.Variable Function : <spirv.ptr <spirv.float 32, Function>>;
        count = spirv.Load count_ptr : <builtin.integer ui32>;
        value = spirv.Load value_ptr, memory_access = VOLATILE : <spirv.float 32>;
        spirv_pliron.loop {
          ^loop_entry():
            spirv.Branch ^header(count)

          ^header(index: builtin.integer ui32):
            less = spirv.ULessThan index, count : <builtin.integer i1>;
            spirv.BranchConditional (less) [^body, ^loop_merge] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>

          ^body():
            square = spirv.FMul value, value : <spirv.float 32> {relaxed_precision, no_contraction};
            spirv.Store value_ptr, square;
            next = spirv.IAdd index, count : <builtin.integer ui32>;
            spirv.Branch ^header(next)

          ^loop_merge():
            spirv_pliron.merge
        };
        spirv.Return
    };
    spirv.EntryPoint GLCompute, @main, "main";
    spirv.ExecutionMode @main, LocalSize, arguments = [1, 1, 1]
}
"#;

/// Parses `input` into `ctx` and verifies the result.
fn parse(ctx: &mut Context, input: &str) -> SpirvModuleOp {
    let op = parse_from_str(Operation::top_level_parser(), ctx, input).unwrap_or_else(|err| panic!("{err}"));
    let module = Operation::get_op::<SpirvModuleOp>(op, ctx).expect("Should be a SPIR-V module");
    verify_op(&module, ctx).unwrap_or_else(|err| panic!("{}", err.disp(ctx)));
    module
}

/// The SPIR-V binary of `module`.
fn emit(ctx: &Context, module: SpirvModuleOp) -> Vec<u32> {
    let mut builder = PlironBuilder::new();
    module.to_spirv(ctx, &mut builder).unwrap();
    builder.module().assemble()
}

/// The parser keeps the name and the requirements of the module.
#[test]
fn module_header_parses() {
    let ctx = &mut Context::new();
    let module = parse(ctx, MODULE);
    assert_eq!(module.get_symbol_name(ctx).to_string(), "module");
    let capabilities = vec![Capability::Shader, Capability::Float16];
    assert_eq!(module.get_vce(ctx), VerCapExtAttr::new((1, 3), capabilities, vec![]));
}

/// The printed form of a parsed module parses to a module with the same binary.
#[test]
fn module_round_trips() {
    let ctx = &mut Context::new();
    let module = parse(ctx, MODULE);
    let printed = module.get_operation().disp(ctx).to_string();

    let reparsed_ctx = &mut Context::new();
    let reparsed = parse(reparsed_ctx, &printed);
    assert_eq!(emit(reparsed_ctx, reparsed), emit(ctx, module));
}
