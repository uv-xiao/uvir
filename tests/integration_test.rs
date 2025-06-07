use uvir::{Context, Value};
use uvir::dialects::builtin::{integer_type, float_type};
use uvir::dialects::arith::{ConstantOp, AddOp, MulOp};
use uvir::attribute::Attribute;
use uvir::FloatPrecision;
use uvir::printer::Printer;

#[test]
fn test_end_to_end_arithmetic_computation() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    // Build an arithmetic computation: (10 + 20) * 3 = 90
    let c1_name = ctx.intern_string("c1");
    let c2_name = ctx.intern_string("c2");
    let c3_name = ctx.intern_string("c3");
    let sum_name = ctx.intern_string("sum");
    let result_name = ctx.intern_string("result");
    
    // Create all values
    let (c1_val, c2_val, c3_val, sum_val, result_val) = {
        let region = ctx.get_global_region_mut();
        
        let c1_val = region.add_value(Value {
            name: Some(c1_name),
            ty: i32_type,
            defining_op: None,
        });
        
        let c2_val = region.add_value(Value {
            name: Some(c2_name),
            ty: i32_type,
            defining_op: None,
        });
        
        let c3_val = region.add_value(Value {
            name: Some(c3_name),
            ty: i32_type,
            defining_op: None,
        });
        
        let sum_val = region.add_value(Value {
            name: Some(sum_name),
            ty: i32_type,
            defining_op: None,
        });
        
        let result_val = region.add_value(Value {
            name: Some(result_name),
            ty: i32_type,
            defining_op: None,
        });
        
        (c1_val, c2_val, c3_val, sum_val, result_val)
    };
    
    // Create operations
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
    
    let add = AddOp {
        result: sum_val,
        lhs: c1_val,
        rhs: c2_val,
    };
    
    let mul = MulOp {
        result: result_val,
        lhs: sum_val,
        rhs: c3_val,
    };
    
    // Add operations in order
    let ops = vec![
        const1.into_op_data(&mut ctx),
        const2.into_op_data(&mut ctx),
        const3.into_op_data(&mut ctx),
        add.into_op_data(&mut ctx),
        mul.into_op_data(&mut ctx),
    ];
    
    {
        let region = ctx.get_global_region_mut();
        for op in ops {
            region.add_op(op);
        }
    }
    
    // Verify the computation structure
    let region = ctx.get_global_region();
    assert_eq!(region.operations.len(), 5);
    assert_eq!(region.values.len(), 5);
    
    // Print the module
    let mut printer = Printer::new();
    printer.print_module(&ctx).unwrap();
    let output = printer.get_output();
    println!("{}", output);
    
    // Verify output contains expected operations
    assert!(output.contains("arith.constant"));
    assert!(output.contains("arith.addi"));
    assert!(output.contains("arith.muli"));
}

#[test]
fn test_mixed_type_operations() {
    let mut ctx = Context::new();
    
    let i32_type = integer_type(&mut ctx, 32, true);
    let i64_type = integer_type(&mut ctx, 64, true);
    let f32_type = float_type(&mut ctx, FloatPrecision::Single);
    let f64_type = float_type(&mut ctx, FloatPrecision::Double);
    
    let i32_val_name = ctx.intern_string("i32_val");
    let i64_val_name = ctx.intern_string("i64_val");
    let f32_val_name = ctx.intern_string("f32_val");
    let f64_val_name = ctx.intern_string("f64_val");
    
    // Create values of different types
    let (i32_val, i64_val, f32_val, f64_val) = {
        let region = ctx.get_global_region_mut();
        
        let i32_val = region.add_value(Value {
            name: Some(i32_val_name),
            ty: i32_type,
            defining_op: None,
        });
        
        let i64_val = region.add_value(Value {
            name: Some(i64_val_name),
            ty: i64_type,
            defining_op: None,
        });
        
        let f32_val = region.add_value(Value {
            name: Some(f32_val_name),
            ty: f32_type,
            defining_op: None,
        });
        
        let f64_val = region.add_value(Value {
            name: Some(f64_val_name),
            ty: f64_type,
            defining_op: None,
        });
        
        (i32_val, i64_val, f32_val, f64_val)
    };
    
    // Create constants of different types
    let const_i32 = ConstantOp {
        result: i32_val,
        value: Attribute::Integer(42),
    };
    
    let const_i64 = ConstantOp {
        result: i64_val,
        value: Attribute::Integer(i64::MAX),
    };
    
    let const_f32 = ConstantOp {
        result: f32_val,
        value: Attribute::Float(3.14),
    };
    
    let const_f64 = ConstantOp {
        result: f64_val,
        value: Attribute::Float(std::f64::consts::E),
    };
    
    // Add all operations
    let ops = vec![
        const_i32.into_op_data(&mut ctx),
        const_i64.into_op_data(&mut ctx),
        const_f32.into_op_data(&mut ctx),
        const_f64.into_op_data(&mut ctx),
    ];
    
    {
        let region = ctx.get_global_region_mut();
        for op in ops {
            region.add_op(op);
        }
    }
    
    // Verify different types were created
    let region = ctx.get_global_region();
    let types: Vec<_> = region.values.values()
        .map(|v| v.ty)
        .collect();
    
    assert_eq!(types.len(), 4);
    assert!(types.contains(&i32_type));
    assert!(types.contains(&i64_type));
    assert!(types.contains(&f32_type));
    assert!(types.contains(&f64_type));
}

#[test]
fn test_complex_expression_building() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    // Build: ((a + b) * c) + (d * e)
    let names: Vec<_> = ["a", "b", "c", "d", "e", "t1", "t2", "t3", "result"]
        .iter()
        .map(|&name| ctx.intern_string(name))
        .collect();
    
    // Create all values
    let values: Vec<_> = {
        let region = ctx.get_global_region_mut();
        names.iter().map(|&name| {
            region.add_value(Value {
                name: Some(name),
                ty: i32_type,
                defining_op: None,
            })
        }).collect()
    };
    
    // Create constants
    let constants = vec![1, 2, 3, 4, 5];
    let const_ops: Vec<_> = constants.iter().enumerate().map(|(i, &val)| {
        let const_op = ConstantOp {
            result: values[i],
            value: Attribute::Integer(val),
        };
        const_op.into_op_data(&mut ctx)
    }).collect();
    
    {
        let region = ctx.get_global_region_mut();
        for op in const_ops {
            region.add_op(op);
        }
    }
    
    // t1 = a + b
    let add1 = AddOp {
        result: values[5],
        lhs: values[0],
        rhs: values[1],
    };
    let add1_data = add1.into_op_data(&mut ctx);
    
    // t2 = t1 * c
    let mul1 = MulOp {
        result: values[6],
        lhs: values[5],
        rhs: values[2],
    };
    let mul1_data = mul1.into_op_data(&mut ctx);
    
    // t3 = d * e
    let mul2 = MulOp {
        result: values[7],
        lhs: values[3],
        rhs: values[4],
    };
    let mul2_data = mul2.into_op_data(&mut ctx);
    
    // result = t2 + t3
    let add2 = AddOp {
        result: values[8],
        lhs: values[6],
        rhs: values[7],
    };
    let add2_data = add2.into_op_data(&mut ctx);
    
    {
        let region = ctx.get_global_region_mut();
        region.add_op(add1_data);
        region.add_op(mul1_data);
        region.add_op(mul2_data);
        region.add_op(add2_data);
    }
    
    // Verify structure: 5 constants + 2 adds + 2 muls = 9 operations
    let region = ctx.get_global_region();
    assert_eq!(region.operations.len(), 9);
    
    // Print and verify
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    let output = printer.get_output();
    
    // Should have the expected number of each operation
    let const_count = output.matches("arith.constant").count();
    let add_count = output.matches("arith.addi").count();
    let mul_count = output.matches("arith.muli").count();
    
    assert_eq!(const_count, 5);
    assert_eq!(add_count, 2);
    assert_eq!(mul_count, 2);
}

#[test]
fn test_operation_with_attributes() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    let val_name = ctx.intern_string("val");
    let metadata_key = ctx.intern_string("metadata");
    let metadata_val = ctx.intern_string("important");
    let flags_key = ctx.intern_string("flags");
    let readonly = ctx.intern_string("readonly");
    let noinline = ctx.intern_string("noinline");
    let priority_key = ctx.intern_string("priority");
    
    // Create a value
    let val = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(val_name),
            ty: i32_type,
            defining_op: None,
        })
    };
    
    // Create operation with custom attributes
    let const_op = ConstantOp {
        result: val,
        value: Attribute::Integer(42),
    };
    
    let mut op_data = const_op.into_op_data(&mut ctx);
    
    // Add custom attributes
    let custom_attrs = vec![
        (metadata_key, Attribute::String(metadata_val)),
        (flags_key, Attribute::Array(vec![
            Attribute::String(readonly),
            Attribute::String(noinline),
        ])),
        (priority_key, Attribute::Integer(10)),
    ];
    
    for (k, v) in custom_attrs {
        op_data.attributes.push((k, v));
    }
    
    {
        let region = ctx.get_global_region_mut();
        region.add_op(op_data);
    }
    
    // Verify attributes were preserved
    let region = ctx.get_global_region();
    let op = region.operations.values().next().unwrap();
    assert!(op.attributes.len() > 0);
    
    // Find the metadata attribute
    let metadata = op.attributes.iter()
        .find(|(k, _)| *k == metadata_key);
    assert!(metadata.is_some());
}

#[test]
fn test_multiple_regions() {
    let mut ctx = Context::new();
    
    // Create additional regions
    let region1 = ctx.create_region();
    let region2 = ctx.create_region();
    
    let i32_type = integer_type(&mut ctx, 32, true);
    let global_val_name = ctx.intern_string("global_val");
    let r1_val_name = ctx.intern_string("r1_val");
    let r2_val_name = ctx.intern_string("r2_val");
    
    // Add operations to different regions
    {
        // Global region
        let val = {
            let global = ctx.get_global_region_mut();
            global.add_value(Value {
                name: Some(global_val_name),
                ty: i32_type,
                defining_op: None,
            })
        };
        
        let const_op = ConstantOp {
            result: val,
            value: Attribute::Integer(1),
        };
        let op_data = const_op.into_op_data(&mut ctx);
        ctx.get_global_region_mut().add_op(op_data);
    }
    
    {
        // Region 1
        let val = {
            let r1 = ctx.get_region_mut(region1).unwrap();
            r1.add_value(Value {
                name: Some(r1_val_name),
                ty: i32_type,
                defining_op: None,
            })
        };
        
        let const_op = ConstantOp {
            result: val,
            value: Attribute::Integer(2),
        };
        let op_data = const_op.into_op_data(&mut ctx);
        ctx.get_region_mut(region1).unwrap().add_op(op_data);
    }
    
    {
        // Region 2
        let val = {
            let r2 = ctx.get_region_mut(region2).unwrap();
            r2.add_value(Value {
                name: Some(r2_val_name),
                ty: i32_type,
                defining_op: None,
            })
        };
        
        let const_op = ConstantOp {
            result: val,
            value: Attribute::Integer(3),
        };
        let op_data = const_op.into_op_data(&mut ctx);
        ctx.get_region_mut(region2).unwrap().add_op(op_data);
    }
    
    // Verify each region has exactly one operation
    assert_eq!(ctx.get_global_region().operations.len(), 1);
    assert_eq!(ctx.get_region(region1).unwrap().operations.len(), 1);
    assert_eq!(ctx.get_region(region2).unwrap().operations.len(), 1);
}

#[test]
fn test_value_use_def_chains() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    // Pre-intern strings
    let val_names: Vec<_> = (0..4).map(|i| ctx.intern_string(&format!("v{}", i))).collect();
    
    // Create a chain of operations with use-def relationships
    let vals: Vec<_> = {
        let region = ctx.get_global_region_mut();
        val_names.iter().map(|&name| {
            region.add_value(Value {
                name: Some(name),
                ty: i32_type,
                defining_op: None,
            })
        }).collect()
    };
    
    // v0 = constant 10
    let const_op = ConstantOp {
        result: vals[0],
        value: Attribute::Integer(10),
    };
    let const_op_data = const_op.into_op_data(&mut ctx);
    let const_ref = ctx.get_global_region_mut().add_op(const_op_data);
    ctx.get_global_region_mut().values.get_mut(vals[0]).unwrap().defining_op = Some(uvir::ops::OpRef(const_ref));
    
    // v1 = v0 + v0
    let add1 = AddOp {
        result: vals[1],
        lhs: vals[0],
        rhs: vals[0],
    };
    let add1_data = add1.into_op_data(&mut ctx);
    let add1_ref = ctx.get_global_region_mut().add_op(add1_data);
    ctx.get_global_region_mut().values.get_mut(vals[1]).unwrap().defining_op = Some(uvir::ops::OpRef(add1_ref));
    
    // v2 = v1 * v0
    let mul = MulOp {
        result: vals[2],
        lhs: vals[1],
        rhs: vals[0],
    };
    let mul_data = mul.into_op_data(&mut ctx);
    let mul_ref = ctx.get_global_region_mut().add_op(mul_data);
    ctx.get_global_region_mut().values.get_mut(vals[2]).unwrap().defining_op = Some(uvir::ops::OpRef(mul_ref));
    
    // v3 = v2 + v1
    let add2 = AddOp {
        result: vals[3],
        lhs: vals[2],
        rhs: vals[1],
    };
    let add2_data = add2.into_op_data(&mut ctx);
    let add2_ref = ctx.get_global_region_mut().add_op(add2_data);
    ctx.get_global_region_mut().values.get_mut(vals[3]).unwrap().defining_op = Some(uvir::ops::OpRef(add2_ref));
    
    // Verify use-def chains
    let region = ctx.get_global_region();
    for (i, &val) in vals.iter().enumerate() {
        let value = region.values.get(val).unwrap();
        assert!(value.defining_op.is_some(), "Value v{} should have defining op", i);
    }
    
    // Verify the last operation uses v2 and v1
    let last_op = region.get_op(add2_ref).unwrap();
    assert_eq!(last_op.operands.len(), 2);
    assert_eq!(last_op.operands[0], vals[2]);
    assert_eq!(last_op.operands[1], vals[1]);
}

#[test]
fn test_large_scale_ir_construction() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    const NUM_OPERATIONS: usize = 100;
    
    // Pre-intern all strings
    let names: Vec<_> = (0..NUM_OPERATIONS * 2)
        .map(|i| ctx.intern_string(&format!("v{}", i)))
        .collect();
    
    // Create many values
    let values: Vec<_> = {
        let region = ctx.get_global_region_mut();
        names.iter().map(|&name| {
            region.add_value(Value {
                name: Some(name),
                ty: i32_type,
                defining_op: None,
            })
        }).collect()
    };
    
    // Create constants for first half
    let const_ops: Vec<_> = (0..NUM_OPERATIONS).map(|i| {
        let const_op = ConstantOp {
            result: values[i],
            value: Attribute::Integer(i as i64),
        };
        const_op.into_op_data(&mut ctx)
    }).collect();
    
    {
        let region = ctx.get_global_region_mut();
        for op in const_ops {
            region.add_op(op);
        }
    }
    
    // Create arithmetic operations using the constants
    let arith_ops: Vec<_> = (0..NUM_OPERATIONS).map(|i| {
        let result_idx = NUM_OPERATIONS + i;
        if i % 2 == 0 {
            // Even: add operation
            AddOp {
                result: values[result_idx],
                lhs: values[i / 2],
                rhs: values[i / 2 + 1],
            }.into_op_data(&mut ctx)
        } else {
            // Odd: multiply operation
            MulOp {
                result: values[result_idx],
                lhs: values[i / 2],
                rhs: values[i / 2 + 1],
            }.into_op_data(&mut ctx)
        }
    }).collect();
    
    {
        let region = ctx.get_global_region_mut();
        for op in arith_ops {
            region.add_op(op);
        }
    }
    
    // Verify scale
    let region = ctx.get_global_region();
    assert_eq!(region.values.len(), NUM_OPERATIONS * 2);
    assert_eq!(region.operations.len(), NUM_OPERATIONS * 2);
    assert_eq!(region.op_order.len(), NUM_OPERATIONS * 2);
    
    // Verify operation types
    let const_count = region.operations.values()
        .filter(|op| op.info.name == "constant")
        .count();
    let add_count = region.operations.values()
        .filter(|op| op.info.name == "addi")
        .count();
    let mul_count = region.operations.values()
        .filter(|op| op.info.name == "muli")
        .count();
    
    assert_eq!(const_count, NUM_OPERATIONS);
    assert_eq!(add_count + mul_count, NUM_OPERATIONS);
}