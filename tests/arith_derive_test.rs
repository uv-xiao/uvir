use uvir::*;
use uvir::dialects::builtin::integer_type;
use uvir::dialects::arith_derive::{AddOp, MulOp, SubOp, CmpIOp, CmpFOp, SelectOp, AddUIExtendedOp};
use uvir::attribute::Attribute;

#[test]
fn test_arith_ops_basic() {
    let mut ctx = Context::new();
    
    // Create some values
    let i32_ty = integer_type(&mut ctx, 32, true);
    let a = ctx.create_value(Some("a"), i32_ty);
    let b = ctx.create_value(Some("b"), i32_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    
    // Test AddOp
    let add_op = AddOp { lhs: a, rhs: b, result };
    let op_data = add_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "arith");
    assert_eq!(op_data.info.name, "addi");
    assert_eq!(op_data.info.traits, &["Commutative"]);
    assert_eq!(op_data.operands.len(), 2);
    assert_eq!(op_data.results.len(), 1);
    
    // Test MulOp
    let mul_op = MulOp { lhs: a, rhs: b, result };
    let mul_data = mul_op.into_op_data(&mut ctx);
    
    assert_eq!(mul_data.info.dialect, "arith");
    assert_eq!(mul_data.info.name, "muli");
    assert_eq!(mul_data.info.traits, &["Commutative"]);
    
    // Test SubOp (not commutative)
    let sub_op = SubOp { lhs: a, rhs: b, result };
    let sub_data = sub_op.into_op_data(&mut ctx);
    
    assert_eq!(sub_data.info.dialect, "arith");
    assert_eq!(sub_data.info.name, "subi");
    assert_eq!(sub_data.info.traits, &[] as &[&str]);
}

#[test]
fn test_print_arith_ops() {
    let mut ctx = Context::new();
    
    // Create some values
    let i32_ty = integer_type(&mut ctx, 32, true);
    let a = ctx.create_value(Some("a"), i32_ty);
    let b = ctx.create_value(Some("b"), i32_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    
    // Create and print an add operation
    let add_op = AddOp { lhs: a, rhs: b, result };
    let op_data = add_op.into_op_data(&mut ctx);
    
    let mut printer = uvir::printer::Printer::new();
    printer.print_operation(&ctx, &op_data).unwrap();
    let output = printer.get_output();
    
    // Check output contains expected format
    assert!(output.contains("arith.addi"));
    assert!(output.contains("%a"));
    assert!(output.contains("%b"));
    assert!(output.contains("%result"));
}

#[test]
fn test_roundtrip_conversion() {
    let mut ctx = Context::new();
    
    // Create some values
    let i32_ty = integer_type(&mut ctx, 32, true);
    let a = ctx.create_value(Some("a"), i32_ty);
    let b = ctx.create_value(Some("b"), i32_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    
    // Create operation
    let add_op = AddOp { lhs: a, rhs: b, result };
    let add_clone = add_op.clone();
    let op_data = add_op.into_op_data(&mut ctx);
    
    // Convert back
    let recovered = AddOp::from_op_data(&op_data, &ctx);
    
    // Check fields match
    assert_eq!(recovered.lhs, add_clone.lhs);
    assert_eq!(recovered.rhs, add_clone.rhs);
    assert_eq!(recovered.result, add_clone.result);
}

#[test]
fn test_comparison_operations() {
    let mut ctx = Context::new();
    
    // Create values
    let i32_ty = integer_type(&mut ctx, 32, true);
    let bool_ty = integer_type(&mut ctx, 1, false);
    let a = ctx.create_value(Some("a"), i32_ty);
    let b = ctx.create_value(Some("b"), i32_ty);
    let result = ctx.create_value(Some("result"), bool_ty);
    
    // Test integer comparison with predicate attribute
    let predicate = Attribute::String(ctx.intern_string("eq")); // equals
    let cmpi_op = CmpIOp { 
        lhs: a, 
        rhs: b, 
        result,
        predicate
    };
    let op_data = cmpi_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "arith");
    assert_eq!(op_data.info.name, "cmpi");
    assert_eq!(op_data.attributes.len(), 1);
    
    // Test float comparison
    let f32_ty = integer_type(&mut ctx, 32, true); // Using as placeholder for float
    let fa = ctx.create_value(Some("fa"), f32_ty);
    let fb = ctx.create_value(Some("fb"), f32_ty);
    let fresult = ctx.create_value(Some("fresult"), bool_ty);
    
    let fpredicate = Attribute::String(ctx.intern_string("oeq")); // ordered equal
    let cmpf_op = CmpFOp {
        lhs: fa,
        rhs: fb,
        result: fresult,
        predicate: fpredicate
    };
    let op_data = cmpf_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.name, "cmpf");
    assert_eq!(op_data.attributes.len(), 1);
}

#[test]
fn test_select_operation() {
    let mut ctx = Context::new();
    
    // Create values
    let i32_ty = integer_type(&mut ctx, 32, true);
    let bool_ty = integer_type(&mut ctx, 1, false);
    let cond = ctx.create_value(Some("cond"), bool_ty);
    let true_val = ctx.create_value(Some("true_val"), i32_ty);
    let false_val = ctx.create_value(Some("false_val"), i32_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    
    let select_op = SelectOp {
        condition: cond,
        true_value: true_val,
        false_value: false_val,
        result
    };
    let op_data = select_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "arith");
    assert_eq!(op_data.info.name, "select");
    assert_eq!(op_data.operands.len(), 3);
    assert_eq!(op_data.results.len(), 1);
}

#[test]
fn test_overflow_operation() {
    let mut ctx = Context::new();
    
    // Create values
    let i32_ty = integer_type(&mut ctx, 32, false); // unsigned
    let bool_ty = integer_type(&mut ctx, 1, false);
    let a = ctx.create_value(Some("a"), i32_ty);
    let b = ctx.create_value(Some("b"), i32_ty);
    let sum = ctx.create_value(Some("sum"), i32_ty);
    let overflow = ctx.create_value(Some("overflow"), bool_ty);
    
    let add_overflow_op = AddUIExtendedOp {
        lhs: a,
        rhs: b,
        sum,
        overflow
    };
    let op_data = add_overflow_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "arith");
    assert_eq!(op_data.info.name, "addui_extended");
    assert_eq!(op_data.operands.len(), 2);
    assert_eq!(op_data.results.len(), 2); // sum and overflow flag
}