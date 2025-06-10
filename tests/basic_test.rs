use uvir::attribute::Attribute;
use uvir::dialects::arith::{AddOp, ConstantOp};
use uvir::dialects::builtin::integer_type;
use uvir::{Context, Value};

#[test]
fn test_basic_operations() {
    let mut ctx = Context::new();

    // Create i32 type
    let i32_type = integer_type(&mut ctx, 32, true);

    // Intern strings first
    let const1_name = ctx.intern_string("const1");
    let const2_name = ctx.intern_string("const2");
    let sum_name = ctx.intern_string("sum");

    // Create values and operations
    let const1_val;
    let const2_val;
    let sum_val;
    let op1;
    let op2;
    let op3;

    {
        let region = ctx.get_global_region_mut();

        // Create two constants
        const1_val = region.add_value(Value {
            name: Some(const1_name),
            ty: i32_type,
            defining_op: None,
        });

        const2_val = region.add_value(Value {
            name: Some(const2_name),
            ty: i32_type,
            defining_op: None,
        });

        // Create an add operation result
        sum_val = region.add_value(Value {
            name: Some(sum_name),
            ty: i32_type,
            defining_op: None,
        });
    }

    let const1_op = ConstantOp {
        result: const1_val,
        value: Attribute::Integer(42),
    };

    let const2_op = ConstantOp {
        result: const2_val,
        value: Attribute::Integer(58),
    };

    let add_op = AddOp {
        result: sum_val,
        lhs: const1_val,
        rhs: const2_val,
    };

    // Add operations to the region
    let const1_op_data = const1_op.into_op_data(&mut ctx);
    let const2_op_data = const2_op.into_op_data(&mut ctx);
    let add_op_data = add_op.into_op_data(&mut ctx);

    {
        let region = ctx.get_global_region_mut();
        op1 = region.add_op(const1_op_data);
        op2 = region.add_op(const2_op_data);
        op3 = region.add_op(add_op_data);
    }

    // Verify the operations were added
    {
        let region = ctx.get_global_region();
        assert_eq!(region.op_order.len(), 3);

        // Verify we can retrieve the operations
        assert!(region.get_op(op1).is_some());
        assert!(region.get_op(op2).is_some());
        assert!(region.get_op(op3).is_some());

        // Verify the operation info
        let op1_data = region.get_op(op1).unwrap();
        assert_eq!(op1_data.info.dialect, "arith");
        assert_eq!(op1_data.info.name, "constant");

        let op3_data = region.get_op(op3).unwrap();
        assert_eq!(op3_data.info.dialect, "arith");
        assert_eq!(op3_data.info.name, "addi");
    }
}

#[test]
fn test_string_interning() {
    let mut ctx = Context::new();

    let id1 = ctx.intern_string("hello");
    let id2 = ctx.intern_string("world");
    let id3 = ctx.intern_string("hello");

    // Same string should get same ID
    assert_eq!(id1, id3);
    assert_ne!(id1, id2);

    // Should be able to retrieve strings
    assert_eq!(ctx.get_string(id1), Some("hello"));
    assert_eq!(ctx.get_string(id2), Some("world"));
}

#[test]
fn test_type_interning() {
    let mut ctx = Context::new();

    let i32_1 = integer_type(&mut ctx, 32, true);
    let i32_2 = integer_type(&mut ctx, 32, true);
    let i64 = integer_type(&mut ctx, 64, true);

    // Same type should get same ID
    assert_eq!(i32_1, i32_2);
    assert_ne!(i32_1, i64);
}
