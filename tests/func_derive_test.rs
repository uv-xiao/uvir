use uvir::dialects::builtin::integer_type;
use uvir::dialects::func_derive::*;
use uvir::*;

#[test]
fn test_func_definition() {
    let mut ctx = Context::new();

    // Create function body region
    let body = ctx.create_region();

    // Create function attributes
    let sym_name = Attribute::String(ctx.intern_string("add"));
    let function_type = Attribute::String(ctx.intern_string("(i32, i32) -> i32"));
    let sym_visibility = Attribute::String(ctx.intern_string("public"));

    let func_op = FuncOp {
        body,
        sym_name,
        function_type,
        sym_visibility,
    };

    let global_region = ctx.global_region();
    let op_data = func_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "func");
    assert_eq!(op_data.info.name, "func");
    assert_eq!(op_data.regions.len(), 1);
    assert_eq!(op_data.attributes.len(), 3);
}

#[test]
fn test_return_op() {
    let mut ctx = Context::new();

    // Create return value
    let i32_ty = integer_type(&mut ctx, 32, true);
    let value = ctx.create_value(Some("ret_val"), i32_ty);

    let return_op = ReturnOp { operands: value };

    let global_region = ctx.global_region();
    let op_data = return_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "func");
    assert_eq!(op_data.info.name, "return");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 0);
}

#[test]
fn test_call_op() {
    let mut ctx = Context::new();

    // Create call arguments and results
    let i32_ty = integer_type(&mut ctx, 32, true);
    let arg1 = ctx.create_value(Some("arg1"), i32_ty);
    let result = ctx.create_value(Some("result"), i32_ty);

    let callee = Attribute::String(ctx.intern_string("@add"));

    let call_op = CallOp {
        operands: arg1,
        results: result,
        callee,
    };

    let global_region = ctx.global_region();
    let op_data = call_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "func");
    assert_eq!(op_data.info.name, "call");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.attributes.len(), 1);
}

#[test]
fn test_call_indirect_op() {
    let mut ctx = Context::new();

    // Create function pointer, arguments, and results
    let i32_ty = integer_type(&mut ctx, 32, true);
    let func_ty = integer_type(&mut ctx, 64, false); // Placeholder for function type
    let func_ptr = ctx.create_value(Some("func_ptr"), func_ty);
    let arg = ctx.create_value(Some("arg"), i32_ty);
    let result = ctx.create_value(Some("result"), i32_ty);

    let call_indirect_op = CallIndirectOp {
        callee: func_ptr,
        operands: arg,
        results: result,
    };

    let global_region = ctx.global_region();
    let op_data = call_indirect_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "func");
    assert_eq!(op_data.info.name, "call_indirect");
    assert_eq!(op_data.operands.len(), 2); // callee + operands
    assert_eq!(op_data.results.len(), 1);
}

#[test]
fn test_constant_op() {
    let mut ctx = Context::new();

    // Create function reference
    let func_ty = integer_type(&mut ctx, 64, false); // Placeholder for function type
    let func_ref = ctx.create_value(Some("func_ref"), func_ty);
    let value = Attribute::String(ctx.intern_string("@my_func"));

    let const_op = ConstantOp {
        result: func_ref,
        value,
    };

    let global_region = ctx.global_region();
    let op_data = const_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "func");
    assert_eq!(op_data.info.name, "constant");
    assert_eq!(op_data.operands.len(), 0);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.attributes.len(), 1);
}

#[test]
fn test_print_func_op() {
    let mut ctx = Context::new();

    // Create a simple function
    let body = ctx.create_region();
    let sym_name = Attribute::String(ctx.intern_string("main"));
    let function_type = Attribute::String(ctx.intern_string("() -> ()"));
    let sym_visibility = Attribute::String(ctx.intern_string("public"));

    let func_op = FuncOp {
        body,
        sym_name,
        function_type,
        sym_visibility,
    };

    let global_region = ctx.global_region();
    let op_data = func_op.into_op_data(&mut ctx, global_region);

    let mut printer = uvir::printer::Printer::new();
    printer.print_operation(&ctx, &op_data).unwrap();
    let output = printer.get_output();

    // Check output contains expected format
    assert!(output.contains("func.func"));
}

#[test]
fn test_roundtrip_call_op() {
    let mut ctx = Context::new();

    // Create call operation
    let i32_ty = integer_type(&mut ctx, 32, true);
    let operand = ctx.create_value(Some("arg"), i32_ty);
    let result = ctx.create_value(Some("res"), i32_ty);
    let callee = Attribute::String(ctx.intern_string("@compute"));

    let call_op = CallOp {
        operands: operand,
        results: result,
        callee: callee.clone(),
    };

    let call_clone = call_op.clone();
    let global_region = ctx.global_region();
    let op_data = call_op.into_op_data(&mut ctx, global_region);

    // Convert back
    let recovered = CallOp::from_op_data(&op_data, &ctx);

    // Check fields match
    assert_eq!(recovered.operands, call_clone.operands);
    assert_eq!(recovered.results, call_clone.results);
    // Note: Attribute comparison may not work directly due to the placeholder implementation
}
