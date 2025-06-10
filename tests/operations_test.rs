// Tests for the operation infrastructure and registry in uvir.
//
// Purpose: Validates the core operation system including:
// - Operation creation and storage in regions
// - Operation registry for dialect operations
// - Static dispatch through OpInfo function pointers
// - Operation metadata (operands, results, attributes)
// - Value use-def relationships
// - Operation ordering preservation
//
// Operations are the fundamental units of computation in the IR,
// and this infrastructure enables efficient, type-safe operation handling.

use slotmap::Key;
use uvir::attribute::Attribute;
use uvir::dialects::arith::{AddOp, ConstantOp, MulOp};
use uvir::dialects::builtin::integer_type;
use uvir::{Context, Value};

#[test]
fn test_operation_creation() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create values
    let val1_name = ctx.intern_string("val1");
    let val2_name = ctx.intern_string("val2");
    let result_name = ctx.intern_string("result");

    // Create values in the region
    let (val1, val2, result) = {
        let region = ctx.get_global_region_mut();
        let val1 = region.add_value(Value {
            name: Some(val1_name),
            ty: i32_type,
            defining_op: None,
        });

        let val2 = region.add_value(Value {
            name: Some(val2_name),
            ty: i32_type,
            defining_op: None,
        });

        let result = region.add_value(Value {
            name: Some(result_name),
            ty: i32_type,
            defining_op: None,
        });
        (val1, val2, result)
    };

    // Create constant operations
    let const1 = ConstantOp {
        result: val1,
        value: Attribute::Integer(42),
    };

    let const2 = ConstantOp {
        result: val2,
        value: Attribute::Integer(58),
    };

    // Create add operation
    let add = AddOp {
        result,
        lhs: val1,
        rhs: val2,
    };

    // Convert to OpData
    let const1_data = const1.into_op_data(&mut ctx);
    let _const2_data = const2.into_op_data(&mut ctx);
    let add_data = add.into_op_data(&mut ctx);

    // Verify operation metadata
    assert_eq!(const1_data.info.dialect, "arith");
    assert_eq!(const1_data.info.name, "constant");
    assert_eq!(add_data.info.dialect, "arith");
    assert_eq!(add_data.info.name, "addi");

    // Verify operands and results
    assert_eq!(const1_data.operands.len(), 0);
    assert_eq!(const1_data.results.len(), 1);
    assert_eq!(const1_data.results[0], val1);

    assert_eq!(add_data.operands.len(), 2);
    assert_eq!(add_data.operands[0], val1);
    assert_eq!(add_data.operands[1], val2);
    assert_eq!(add_data.results.len(), 1);
    assert_eq!(add_data.results[0], result);
}

#[test]
fn test_operation_registry() {
    let mut ctx = Context::new();

    let arith_dialect = ctx.intern_string("arith");
    let constant_name = ctx.intern_string("constant");
    let add_name = ctx.intern_string("addi");
    let mul_name = ctx.intern_string("muli");
    let fake_name = ctx.intern_string("fake_op");

    let registry = ctx.op_registry();

    // Check that operations are registered
    assert!(registry.get(arith_dialect, constant_name).is_some());
    assert!(registry.get(arith_dialect, add_name).is_some());
    assert!(registry.get(arith_dialect, mul_name).is_some());

    // Check operation info
    let const_info = registry.get(arith_dialect, constant_name).unwrap();
    assert_eq!(const_info.dialect, "arith");
    assert_eq!(const_info.name, "constant");

    // Check non-existent operation
    assert!(registry.get(arith_dialect, fake_name).is_none());
}

#[test]
fn test_operation_storage() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create value first
    let val = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: None,
            ty: i32_type,
            defining_op: None,
        })
    };

    // Create operation
    let const_op = ConstantOp {
        result: val,
        value: Attribute::Integer(123),
    };

    let op_data = const_op.into_op_data(&mut ctx);

    // Add operation to region and verify
    let op_ref = {
        let region = ctx.get_global_region_mut();
        region.add_op(op_data)
    };

    // Retrieve and verify
    let region = ctx.get_global_region();
    let retrieved = region.get_op(op_ref).unwrap();
    assert_eq!(retrieved.info.name, "constant");

    // Try to extract back as ConstantOp
    let extracted = ConstantOp::from_op_data(retrieved);
    assert!(extracted.is_some());
    let extracted = extracted.unwrap();
    assert_eq!(extracted.result, val);
    match &extracted.value {
        Attribute::Integer(v) => assert_eq!(*v, 123),
        _ => panic!("Expected integer attribute"),
    }
}

#[test]
fn test_multiply_operation() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let a_name = ctx.intern_string("a");
    let b_name = ctx.intern_string("b");
    let product_name = ctx.intern_string("product");

    let (a, b, product) = {
        let region = ctx.get_global_region_mut();
        let a = region.add_value(Value {
            name: Some(a_name),
            ty: i32_type,
            defining_op: None,
        });

        let b = region.add_value(Value {
            name: Some(b_name),
            ty: i32_type,
            defining_op: None,
        });

        let product = region.add_value(Value {
            name: Some(product_name),
            ty: i32_type,
            defining_op: None,
        });
        (a, b, product)
    };

    // Create multiply operation
    let mul = MulOp {
        result: product,
        lhs: a,
        rhs: b,
    };

    let mul_data = mul.into_op_data(&mut ctx);

    // Verify
    assert_eq!(mul_data.info.dialect, "arith");
    assert_eq!(mul_data.info.name, "muli");
    assert_eq!(mul_data.operands.len(), 2);
    assert_eq!(mul_data.results.len(), 1);
}

#[test]
fn test_operation_ordering() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Pre-intern strings
    let val_names: Vec<_> = (0..5)
        .map(|i| ctx.intern_string(&format!("val{}", i)))
        .collect();

    // Create multiple operations
    let mut ops = Vec::new();
    for i in 0..5 {
        let val = {
            let region = ctx.get_global_region_mut();
            region.add_value(Value {
                name: Some(val_names[i]),
                ty: i32_type,
                defining_op: None,
            })
        };

        let const_op = ConstantOp {
            result: val,
            value: Attribute::Integer(i as i64),
        };

        let op_data = const_op.into_op_data(&mut ctx);
        let op_ref = {
            let region = ctx.get_global_region_mut();
            region.add_op(op_data)
        };
        ops.push(op_ref);
    }

    // Verify order is preserved
    let region = ctx.get_global_region();
    assert_eq!(region.op_order.len(), 5);
    for (i, &op_ref) in region.op_order.iter().enumerate() {
        assert_eq!(op_ref, ops[i]);
    }
}

#[test]
fn test_operation_attributes() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let attr_key = ctx.intern_string("custom_attr");
    let test_value = ctx.intern_string("test_value");

    let val = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: None,
            ty: i32_type,
            defining_op: None,
        })
    };

    // Create operation with attributes
    let const_op = ConstantOp {
        result: val,
        value: Attribute::Integer(42),
    };

    let mut op_data = const_op.into_op_data(&mut ctx);

    // Add custom attributes
    let attr_val = Attribute::String(test_value);
    op_data.attributes.push((attr_key, attr_val));

    let op_ref = ctx.get_global_region_mut().add_op(op_data);

    // Retrieve and check attributes
    let region = ctx.get_global_region();
    let retrieved = region.get_op(op_ref).unwrap();
    assert!(!retrieved.attributes.is_empty());

    let found = retrieved.attributes.iter().find(|(k, _)| *k == attr_key);
    assert!(found.is_some());
    match &found.unwrap().1 {
        Attribute::String(s) => assert_eq!(*s, test_value),
        _ => panic!("Expected string attribute"),
    }
}

#[test]
fn test_value_defining_op() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let result_name = ctx.intern_string("result");

    // Create a value that will be defined by an operation
    let result = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(result_name),
            ty: i32_type,
            defining_op: None,
        })
    };

    // Create operation that defines the value
    let const_op = ConstantOp {
        result,
        value: Attribute::Integer(100),
    };

    let op_data = const_op.into_op_data(&mut ctx);

    // Add operation and update defining op
    let op_ref = {
        let region = ctx.get_global_region_mut();
        let op_ref = region.add_op(op_data);
        // Update the value's defining op
        region.values.get_mut(result).unwrap().defining_op = Some(uvir::ops::OpRef(op_ref));
        op_ref
    };

    // Verify the relationship
    let region = ctx.get_global_region();
    let val = region.values.get(result).unwrap();
    match val.defining_op {
        Some(uvir::ops::OpRef(opr)) => assert_eq!(opr, op_ref),
        None => panic!("Expected defining op to be set"),
    }

    let op = region.get_op(op_ref).unwrap();
    assert_eq!(op.results[0], result);
}

#[test]
fn test_complex_operation_chain() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let c1_name = ctx.intern_string("c1");
    let c2_name = ctx.intern_string("c2");
    let c3_name = ctx.intern_string("c3");
    let sum_name = ctx.intern_string("sum");
    let product_name = ctx.intern_string("product");

    // Create all values first
    let (c1, c2, c3, sum, product) = {
        let region = ctx.get_global_region_mut();

        let c1 = region.add_value(Value {
            name: Some(c1_name),
            ty: i32_type,
            defining_op: None,
        });

        let c2 = region.add_value(Value {
            name: Some(c2_name),
            ty: i32_type,
            defining_op: None,
        });

        let c3 = region.add_value(Value {
            name: Some(c3_name),
            ty: i32_type,
            defining_op: None,
        });

        let sum = region.add_value(Value {
            name: Some(sum_name),
            ty: i32_type,
            defining_op: None,
        });

        let product = region.add_value(Value {
            name: Some(product_name),
            ty: i32_type,
            defining_op: None,
        });

        (c1, c2, c3, sum, product)
    };

    // Create operations: (c1 + c2) * c3
    let const1 = ConstantOp {
        result: c1,
        value: Attribute::Integer(10),
    };

    let const2 = ConstantOp {
        result: c2,
        value: Attribute::Integer(20),
    };

    let const3 = ConstantOp {
        result: c3,
        value: Attribute::Integer(3),
    };

    let add = AddOp {
        result: sum,
        lhs: c1,
        rhs: c2,
    };

    let mul = MulOp {
        result: product,
        lhs: sum,
        rhs: c3,
    };

    // Convert to OpData
    let const1_data = const1.into_op_data(&mut ctx);
    let const2_data = const2.into_op_data(&mut ctx);
    let const3_data = const3.into_op_data(&mut ctx);
    let add_data = add.into_op_data(&mut ctx);
    let mul_data = mul.into_op_data(&mut ctx);

    // Add all operations
    {
        let region = ctx.get_global_region_mut();
        region.add_op(const1_data);
        region.add_op(const2_data);
        region.add_op(const3_data);
        region.add_op(add_data);
        region.add_op(mul_data);
    }

    // Verify we have all 5 operations
    let region = ctx.get_global_region();
    assert_eq!(region.op_order.len(), 5);

    // Verify the last operation is multiplication
    let last_op = region.get_op(*region.op_order.last().unwrap()).unwrap();
    assert_eq!(last_op.info.name, "muli");
}

#[test]
fn test_operation_null_key_handling() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let val = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: None,
            ty: i32_type,
            defining_op: None,
        })
    };

    let const_op = ConstantOp {
        result: val,
        value: Attribute::Integer(0),
    };

    let op_data = const_op.into_op_data(&mut ctx);
    let op_ref = {
        let region = ctx.get_global_region_mut();
        region.add_op(op_data)
    };

    // Test null key
    let null_op = uvir::ops::Opr::null();
    let region = ctx.get_global_region();
    assert!(region.get_op(null_op).is_none());
    assert_ne!(op_ref, null_op);
}
