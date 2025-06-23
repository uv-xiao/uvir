// Tests for the arithmetic dialect operations in uvir.
//
// Purpose: Validates the arithmetic dialect implementation including:
// - Constant operations for integer and float literals
// - Addition operations (addi) for integer arithmetic
// - Multiplication operations (muli) for integer arithmetic
// - Complex arithmetic expressions and operation chaining
// - Mixed type constants and operations
// - Static operation info and proper dialect registration
//
// The arithmetic dialect provides fundamental computation operations
// and serves as a reference implementation for dialect development.

use uvir::attribute::Attribute;
use uvir::dialects::arith::{AddOp, ConstantOp, MulOp};
use uvir::dialects::builtin::{float_type, integer_type};
use uvir::FloatPrecision;
use uvir::{Context, Value};

#[test]
fn test_constant_integer_operations() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    let i64_type = integer_type(&mut ctx, 64, true);

    let const_i32_name = ctx.intern_string("const_i32");
    let const_i64_name = ctx.intern_string("const_i64");

    // Test i32 constant
    let val_i32 = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(const_i32_name),
            ty: i32_type,
            defining_op: None,
        })
    };

    let const_i32 = ConstantOp {
        result: val_i32,
        value: Attribute::Integer(42),
    };

    let global_region = ctx.global_region();
    let op_data = const_i32.into_op_data(&mut ctx, global_region);
    assert_eq!(op_data.info.dialect, "arith");
    assert_eq!(op_data.info.name, "constant");
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.operands.len(), 0);

    // Test i64 constant
    let val_i64 = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(const_i64_name),
            ty: i64_type,
            defining_op: None,
        })
    };

    let const_i64 = ConstantOp {
        result: val_i64,
        value: Attribute::Integer(i64::MAX),
    };

    let op_data_i64 = const_i64.into_op_data(&mut ctx, global_region);

    {
        let region = ctx.get_global_region_mut();
        region.add_op(op_data);

        // Verify we can extract the constant back before adding
        let extracted = ConstantOp::from_op_data(&op_data_i64).unwrap();
        match &extracted.value {
            Attribute::Integer(v) => assert_eq!(*v, i64::MAX),
            _ => panic!("Expected integer attribute"),
        }

        region.add_op(op_data_i64);
    }
}

#[test]
fn test_constant_float_operations() {
    let mut ctx = Context::new();
    let f32_type = float_type(&mut ctx, FloatPrecision::Single);
    let f64_type = float_type(&mut ctx, FloatPrecision::Double);

    let const_f32_name = ctx.intern_string("const_f32");
    let const_f64_name = ctx.intern_string("const_f64");

    // Test f32 constant
    let val_f32 = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(const_f32_name),
            ty: f32_type,
            defining_op: None,
        })
    };

    let const_f32 = ConstantOp {
        result: val_f32,
        value: Attribute::Float(3.14),
    };

    let global_region = ctx.global_region();
    let op_data_f32 = const_f32.into_op_data(&mut ctx, global_region);

    // Test f64 constant
    let val_f64 = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(const_f64_name),
            ty: f64_type,
            defining_op: None,
        })
    };

    let const_f64 = ConstantOp {
        result: val_f64,
        value: Attribute::Float(std::f64::consts::PI),
    };

    let op_data_f64 = const_f64.into_op_data(&mut ctx, global_region);

    {
        let region = ctx.get_global_region_mut();
        region.add_op(op_data_f32);
        region.add_op(op_data_f64);
    }

    let region = ctx.get_global_region();
    assert_eq!(region.operations.len(), 2);
}

#[test]
fn test_add_operation() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let a_name = ctx.intern_string("a");
    let b_name = ctx.intern_string("b");
    let sum_name = ctx.intern_string("sum");

    // Create operands
    let (a, b, sum) = {
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

        let sum = region.add_value(Value {
            name: Some(sum_name),
            ty: i32_type,
            defining_op: None,
        });

        (a, b, sum)
    };

    // Create add operation
    let add = AddOp {
        result: sum,
        lhs: a,
        rhs: b,
    };

    let global_region = ctx.global_region();
    let add_data = add.into_op_data(&mut ctx, global_region);

    // Verify operation properties
    assert_eq!(add_data.info.dialect, "arith");
    assert_eq!(add_data.info.name, "addi");
    assert_eq!(add_data.operands.len(), 2);
    assert_eq!(add_data.operands[0].val, a);
    assert_eq!(add_data.operands[1].val, b);
    assert_eq!(add_data.results.len(), 1);
    assert_eq!(add_data.results[0], sum);

    // Verify we can extract it back
    let extracted = AddOp::from_op_data(&add_data).unwrap();
    assert_eq!(extracted.result, sum);
    assert_eq!(extracted.lhs, a);
    assert_eq!(extracted.rhs, b);
}

#[test]
fn test_multiply_operation() {
    let mut ctx = Context::new();
    let i64_type = integer_type(&mut ctx, 64, true);

    let x_name = ctx.intern_string("x");
    let y_name = ctx.intern_string("y");
    let product_name = ctx.intern_string("product");

    // Create operands
    let (x, y, product) = {
        let region = ctx.get_global_region_mut();

        let x = region.add_value(Value {
            name: Some(x_name),
            ty: i64_type,
            defining_op: None,
        });

        let y = region.add_value(Value {
            name: Some(y_name),
            ty: i64_type,
            defining_op: None,
        });

        let product = region.add_value(Value {
            name: Some(product_name),
            ty: i64_type,
            defining_op: None,
        });

        (x, y, product)
    };

    // Create multiply operation
    let mul = MulOp {
        result: product,
        lhs: x,
        rhs: y,
    };

    let global_region = ctx.global_region();
    let mul_data = mul.into_op_data(&mut ctx, global_region);

    // Verify operation properties
    assert_eq!(mul_data.info.dialect, "arith");
    assert_eq!(mul_data.info.name, "muli");
    assert_eq!(mul_data.operands.len(), 2);
    assert_eq!(mul_data.results.len(), 1);

    // Verify extraction
    let extracted = MulOp::from_op_data(&mul_data).unwrap();
    assert_eq!(extracted.result, product);
    assert_eq!(extracted.lhs, x);
    assert_eq!(extracted.rhs, y);
}

#[test]
fn test_arithmetic_expression() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Build expression: (10 + 20) * 3 = 90
    let c1_name = ctx.intern_string("c1");
    let c2_name = ctx.intern_string("c2");
    let c3_name = ctx.intern_string("c3");
    let sum_name = ctx.intern_string("sum");
    let result_name = ctx.intern_string("result");

    let (c1_val, c2_val, c3_val, sum_val, result_val) = {
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

        let result = region.add_value(Value {
            name: Some(result_name),
            ty: i32_type,
            defining_op: None,
        });

        (c1, c2, c3, sum, result)
    };

    // Create constants
    let const1 = ConstantOp {
        result: c1_val,
        value: Attribute::Integer(10),
    };

    let const2 = ConstantOp {
        result: c2_val,
        value: Attribute::Integer(20),
    };

    let const3 = ConstantOp {
        result: c3_val,
        value: Attribute::Integer(3),
    };

    // Create add: c1 + c2 = sum
    let add = AddOp {
        result: sum_val,
        lhs: c1_val,
        rhs: c2_val,
    };

    // Create multiply: sum * c3 = result
    let mul = MulOp {
        result: result_val,
        lhs: sum_val,
        rhs: c3_val,
    };

    // Convert to OpData and add to region
    let global_region = ctx.global_region();
    let ops = vec![
        const1.into_op_data(&mut ctx, global_region),
        const2.into_op_data(&mut ctx, global_region),
        const3.into_op_data(&mut ctx, global_region),
        add.into_op_data(&mut ctx, global_region),
        mul.into_op_data(&mut ctx, global_region),
    ];

    {
        let region = ctx.get_global_region_mut();
        for op in ops {
            region.add_op(op);
        }
    }

    // Verify the expression was built correctly
    let region = ctx.get_global_region();
    assert_eq!(region.operations.len(), 5);
    assert_eq!(region.op_order.len(), 5);
}

#[test]
fn test_mixed_type_constants() {
    let mut ctx = Context::new();

    // Create various types
    let i8_type = integer_type(&mut ctx, 8, true);
    let u32_type = integer_type(&mut ctx, 32, false);
    let f16_type = float_type(&mut ctx, FloatPrecision::Half);
    let str_type = integer_type(&mut ctx, 1, true); // Dummy for string attrs

    let hello_str = ctx.intern_string("hello");

    // Create values
    let (int_val, uint_val, float_val, str_val) = {
        let region = ctx.get_global_region_mut();

        // Integer constant
        let int_val = region.add_value(Value {
            name: None,
            ty: i8_type,
            defining_op: None,
        });

        // Unsigned constant
        let uint_val = region.add_value(Value {
            name: None,
            ty: u32_type,
            defining_op: None,
        });

        // Float constant
        let float_val = region.add_value(Value {
            name: None,
            ty: f16_type,
            defining_op: None,
        });

        // String constant (using string attribute)
        let str_val = region.add_value(Value {
            name: None,
            ty: str_type,
            defining_op: None,
        });

        (int_val, uint_val, float_val, str_val)
    };

    let int_const = ConstantOp {
        result: int_val,
        value: Attribute::Integer(-128),
    };

    let uint_const = ConstantOp {
        result: uint_val,
        value: Attribute::Integer(4294967295), // max u32
    };

    let float_const = ConstantOp {
        result: float_val,
        value: Attribute::Float(65504.0), // max f16
    };

    let str_const = ConstantOp {
        result: str_val,
        value: Attribute::String(hello_str),
    };

    // Add all operations
    let global_region = ctx.global_region();
    let ops = vec![
        int_const.into_op_data(&mut ctx, global_region),
        uint_const.into_op_data(&mut ctx, global_region),
        float_const.into_op_data(&mut ctx, global_region),
        str_const.into_op_data(&mut ctx, global_region),
    ];

    {
        let region = ctx.get_global_region_mut();
        for op in ops {
            region.add_op(op);
        }
    }

    let region = ctx.get_global_region();
    assert_eq!(region.operations.len(), 4);
}

#[test]
fn test_chained_additions() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create chain: a + b + c + d
    let a_name = ctx.intern_string("a");
    let b_name = ctx.intern_string("b");
    let c_name = ctx.intern_string("c");
    let d_name = ctx.intern_string("d");
    let sum1_name = ctx.intern_string("sum1");
    let sum2_name = ctx.intern_string("sum2");
    let final_sum_name = ctx.intern_string("final_sum");

    let values = {
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

        let c = region.add_value(Value {
            name: Some(c_name),
            ty: i32_type,
            defining_op: None,
        });

        let d = region.add_value(Value {
            name: Some(d_name),
            ty: i32_type,
            defining_op: None,
        });

        let sum1 = region.add_value(Value {
            name: Some(sum1_name),
            ty: i32_type,
            defining_op: None,
        });

        let sum2 = region.add_value(Value {
            name: Some(sum2_name),
            ty: i32_type,
            defining_op: None,
        });

        let final_sum = region.add_value(Value {
            name: Some(final_sum_name),
            ty: i32_type,
            defining_op: None,
        });

        (a, b, c, d, sum1, sum2, final_sum)
    };

    let (a, b, c, d, sum1, sum2, final_sum) = values;

    // Create constants
    let global_region = ctx.global_region();
    let ops = vec![
        ConstantOp {
            result: a,
            value: Attribute::Integer(1),
        }
        .into_op_data(&mut ctx, global_region),
        ConstantOp {
            result: b,
            value: Attribute::Integer(2),
        }
        .into_op_data(&mut ctx, global_region),
        ConstantOp {
            result: c,
            value: Attribute::Integer(3),
        }
        .into_op_data(&mut ctx, global_region),
        ConstantOp {
            result: d,
            value: Attribute::Integer(4),
        }
        .into_op_data(&mut ctx, global_region),
        AddOp {
            result: sum1,
            lhs: a,
            rhs: b,
        }
        .into_op_data(&mut ctx, global_region), // 1 + 2 = 3
        AddOp {
            result: sum2,
            lhs: sum1,
            rhs: c,
        }
        .into_op_data(&mut ctx, global_region), // 3 + 3 = 6
        AddOp {
            result: final_sum,
            lhs: sum2,
            rhs: d,
        }
        .into_op_data(&mut ctx, global_region), // 6 + 4 = 10
    ];

    {
        let region = ctx.get_global_region_mut();
        for op in ops {
            region.add_op(op);
        }
    }

    let region = ctx.get_global_region();
    assert_eq!(region.operations.len(), 7);
}

#[test]
fn test_operation_info_static() {
    // Verify that operation info is truly static
    let info1 = &ConstantOp::INFO;
    let info2 = &ConstantOp::INFO;

    // Should be the same pointer
    assert_eq!(info1 as *const _, info2 as *const _);

    // Verify info contents
    assert_eq!(ConstantOp::INFO.dialect, "arith");
    assert_eq!(ConstantOp::INFO.name, "constant");

    assert_eq!(AddOp::INFO.dialect, "arith");
    assert_eq!(AddOp::INFO.name, "addi");

    assert_eq!(MulOp::INFO.dialect, "arith");
    assert_eq!(MulOp::INFO.name, "muli");
}
