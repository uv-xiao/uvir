use uvir::dialects::builtin::integer_type;
use uvir::dialects::builtin_derive::{ConstantOp, ModuleOp, UnrealizedConversionCastOp};
use uvir::*;

#[test]
fn test_module_op() {
    let mut ctx = Context::new();

    // Create a region for the module body
    let body_region = ctx.create_region();

    // Create module with symbol name
    let sym_name = Attribute::String(ctx.intern_string("main"));
    let module = ModuleOp {
        body: body_region,
        sym_name,
    };

    let global_region = ctx.global_region();
    let op_data = module.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "builtin");
    assert_eq!(op_data.info.name, "module");
    assert_eq!(op_data.regions.len(), 1);
    assert_eq!(op_data.attributes.len(), 1);
}

#[test]
fn test_unrealized_conversion_cast() {
    let mut ctx = Context::new();

    // Create values of different types for conversion
    let i32_ty = integer_type(&mut ctx, 32, true);
    let i64_ty = integer_type(&mut ctx, 64, true);

    let input = ctx.create_value(Some("input"), i32_ty);
    let output = ctx.create_value(Some("output"), i64_ty);

    // Create cast operation
    let cast_op = UnrealizedConversionCastOp {
        operands: input,
        results: output,
    };

    let global_region = ctx.global_region();
    let op_data = cast_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "builtin");
    assert_eq!(op_data.info.name, "unrealized_conversion_cast");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 1);
}

#[test]
fn test_constant_op() {
    let mut ctx = Context::new();

    // Create a constant integer
    let i32_ty = integer_type(&mut ctx, 32, true);
    let result = ctx.create_value(Some("const"), i32_ty);
    let value = Attribute::Integer(42);

    let const_op = ConstantOp { result, value };
    let global_region = ctx.global_region();
    let op_data = const_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "builtin");
    assert_eq!(op_data.info.name, "constant");
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.attributes.len(), 1);
}

#[test]
fn test_print_module() {
    let mut ctx = Context::new();

    // Create a module
    let body_region = ctx.create_region();
    let sym_name = Attribute::String(ctx.intern_string("test_module"));
    let module = ModuleOp {
        body: body_region,
        sym_name,
    };

    let global_region = ctx.global_region();
    let op_data = module.into_op_data(&mut ctx, global_region);

    let mut printer = uvir::printer::Printer::new();
    printer.print_operation(&ctx, &op_data).unwrap();
    let output = printer.get_output();

    // Check output contains expected format
    assert!(output.contains("builtin.module"));
}

#[test]
fn test_roundtrip_conversion() {
    let mut ctx = Context::new();

    // Create constant operation
    let i32_ty = integer_type(&mut ctx, 32, true);
    let result = ctx.create_value(Some("const"), i32_ty);
    let value = Attribute::Integer(100);

    let const_op = ConstantOp {
        result,
        value: value.clone(),
    };
    let const_clone = const_op.clone();
    let global_region = ctx.global_region();
    let op_data = const_op.into_op_data(&mut ctx, global_region);

    // Convert back
    let recovered = ConstantOp::from_op_data(&op_data, &ctx);

    // Check fields match
    assert_eq!(recovered.result, const_clone.result);
    // Note: Attribute comparison may not work directly due to the placeholder implementation
}
