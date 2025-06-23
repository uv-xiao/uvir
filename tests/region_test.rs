// Tests for region and value management in uvir.
//
// Purpose: Validates the region system functionality including:
// - Region creation and management (global and additional regions)
// - Value storage and retrieval within regions
// - Operation ordering preservation in regions
// - Region isolation (values/ops in one region don't affect others)
// - Large-scale region operations for performance validation
// - Null key handling for safety
//
// Regions provide hierarchical scoping for operations and values,
// essential for representing nested constructs and isolated computations.

use slotmap::Key;
use uvir::attribute::Attribute;
use uvir::dialects::arith::ConstantOp;
use uvir::dialects::builtin::integer_type;
use uvir::region::RegionManager;
use uvir::{Context, Value};

#[test]
fn test_region_creation() {
    let ctx = Context::new();

    // Test global region exists
    let global_region = ctx.get_global_region();
    assert_eq!(global_region.values.len(), 0);
    assert_eq!(global_region.operations.len(), 0);
    assert_eq!(global_region.op_order.len(), 0);
}

#[test]
fn test_value_management() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    let i64_type = integer_type(&mut ctx, 64, true);

    let name1 = ctx.intern_string("value1");
    let name2 = ctx.intern_string("value2");

    let region = ctx.get_global_region_mut();

    // Add values
    let val1 = region.add_value(Value {
        name: Some(name1),
        ty: i32_type,
        defining_op: None,
    });

    let val2 = region.add_value(Value {
        name: Some(name2),
        ty: i64_type,
        defining_op: None,
    });

    // Verify values were added
    assert_eq!(region.values.len(), 2);

    // Retrieve values
    let retrieved1 = region.values.get(val1).unwrap();
    assert_eq!(retrieved1.name, Some(name1));
    assert_eq!(retrieved1.ty, i32_type);
    assert!(retrieved1.defining_op.is_none());

    let retrieved2 = region.values.get(val2).unwrap();
    assert_eq!(retrieved2.name, Some(name2));
    assert_eq!(retrieved2.ty, i64_type);
}

#[test]
fn test_anonymous_values() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let region = ctx.get_global_region_mut();

    // Create anonymous values (no name)
    let anon1 = region.add_value(Value {
        name: None,
        ty: i32_type,
        defining_op: None,
    });

    let anon2 = region.add_value(Value {
        name: None,
        ty: i32_type,
        defining_op: None,
    });

    // Verify they are different values
    assert_ne!(anon1, anon2);

    // Both should have no name
    assert_eq!(region.values.get(anon1).unwrap().name, None);
    assert_eq!(region.values.get(anon2).unwrap().name, None);
}

#[test]
fn test_operation_order_preservation() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create multiple operations
    let mut op_refs = Vec::new();
    for i in 0..10 {
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
            value: Attribute::Integer(i),
        };

        let global_region = ctx.global_region();
        let op_data = const_op.into_op_data(&mut ctx, global_region);
        let op_ref = {
            let region = ctx.get_global_region_mut();
            region.add_op(op_data)
        };
        op_refs.push(op_ref);
    }

    // Verify order is preserved
    let region = ctx.get_global_region();
    assert_eq!(region.op_order.len(), 10);
    for (i, &op_ref) in region.op_order.iter().enumerate() {
        assert_eq!(op_ref, op_refs[i]);
    }
}

#[test]
fn test_multiple_regions() {
    let mut ctx = Context::new();

    // Create additional regions
    let region1 = ctx.create_region();
    let region2 = ctx.create_region();

    // Verify they are different
    assert_ne!(region1, region2);
    assert_ne!(region1, ctx.global_region());
    assert_ne!(region2, ctx.global_region());

    // Get regions and verify they're empty
    let r1 = ctx.get_region(region1).unwrap();
    assert_eq!(r1.values.len(), 0);
    assert_eq!(r1.operations.len(), 0);

    let r2 = ctx.get_region(region2).unwrap();
    assert_eq!(r2.values.len(), 0);
    assert_eq!(r2.operations.len(), 0);
}

#[test]
fn test_region_isolation() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create a new region
    let region_id = ctx.create_region();

    // Add values to global region
    let global_name = ctx.intern_string("global_val");
    {
        let global = ctx.get_global_region_mut();
        global.add_value(Value {
            name: Some(global_name),
            ty: i32_type,
            defining_op: None,
        });
    }

    // Add values to new region
    let region_name = ctx.intern_string("region_val");
    {
        let region = ctx.get_region_mut(region_id).unwrap();
        region.add_value(Value {
            name: Some(region_name),
            ty: i32_type,
            defining_op: None,
        });
    }

    // Verify isolation
    assert_eq!(ctx.get_global_region().values.len(), 1);
    assert_eq!(ctx.get_region(region_id).unwrap().values.len(), 1);
}

#[test]
fn test_null_value_handling() {
    let ctx = Context::new();
    let region = ctx.get_global_region();

    // Test null value key
    let null_val = uvir::ops::Val::null();
    assert!(region.values.get(null_val).is_none());
}

#[test]
fn test_region_with_mixed_content() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    let f32_type = uvir::dialects::builtin::float_type(&mut ctx, uvir::FloatPrecision::Single);

    let int_val_name = ctx.intern_string("int_val");
    let float_val_name = ctx.intern_string("float_val");

    let (int_val, float_val) = {
        let region = ctx.get_global_region_mut();

        // Add different types of values
        let int_val = region.add_value(Value {
            name: Some(int_val_name),
            ty: i32_type,
            defining_op: None,
        });

        let float_val = region.add_value(Value {
            name: Some(float_val_name),
            ty: f32_type,
            defining_op: None,
        });

        (int_val, float_val)
    };

    // Add operations using these values
    let const_int = ConstantOp {
        result: int_val,
        value: Attribute::Integer(42),
    };

    let const_float = ConstantOp {
        result: float_val,
        value: Attribute::Float(3.14),
    };

    let global_region = ctx.global_region();
    let int_op_data = const_int.into_op_data(&mut ctx, global_region);
    let global_region = ctx.global_region();
    let float_op_data = const_float.into_op_data(&mut ctx, global_region);

    {
        let region = ctx.get_global_region_mut();
        region.add_op(int_op_data);
        region.add_op(float_op_data);
    }

    // Verify mixed content
    let region = ctx.get_global_region();
    assert_eq!(region.values.len(), 2);
    assert_eq!(region.operations.len(), 2);
    assert_eq!(region.op_order.len(), 2);
}

#[test]
fn test_value_type_retrieval() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    let i64_type = integer_type(&mut ctx, 64, true);
    let f32_type = uvir::dialects::builtin::float_type(&mut ctx, uvir::FloatPrecision::Single);

    let region = ctx.get_global_region_mut();

    // Create values with different types
    let vals = vec![
        (
            region.add_value(Value {
                name: None,
                ty: i32_type,
                defining_op: None,
            }),
            i32_type,
        ),
        (
            region.add_value(Value {
                name: None,
                ty: i64_type,
                defining_op: None,
            }),
            i64_type,
        ),
        (
            region.add_value(Value {
                name: None,
                ty: f32_type,
                defining_op: None,
            }),
            f32_type,
        ),
    ];

    // Verify types are correctly stored
    for (val, expected_type) in vals {
        let value = region.values.get(val).unwrap();
        assert_eq!(value.ty, expected_type);
    }
}

#[test]
fn test_large_region() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Add many values and operations
    const NUM_OPS: usize = 1000;
    let mut values = Vec::new();

    // Pre-intern all strings
    let names: Vec<_> = (0..NUM_OPS)
        .map(|i| ctx.intern_string(&format!("v{}", i)))
        .collect();

    for i in 0..NUM_OPS {
        let val = {
            let region = ctx.get_global_region_mut();
            region.add_value(Value {
                name: Some(names[i]),
                ty: i32_type,
                defining_op: None,
            })
        };
        values.push(val);

        let const_op = ConstantOp {
            result: val,
            value: Attribute::Integer(i as i64),
        };

        let global_region = ctx.global_region();
        let op_data = const_op.into_op_data(&mut ctx, global_region);
        ctx.get_global_region_mut().add_op(op_data);
    }

    // Verify everything was added
    let region = ctx.get_global_region();
    assert_eq!(region.values.len(), NUM_OPS);
    assert_eq!(region.operations.len(), NUM_OPS);
    assert_eq!(region.op_order.len(), NUM_OPS);

    // Verify we can retrieve random values
    for i in (0..NUM_OPS).step_by(100) {
        let val = region.values.get(values[i]).unwrap();
        assert_eq!(val.name, Some(names[i]));
    }
}

#[test]
fn test_region_manager() {
    let mut manager = RegionManager::new();

    // Initially empty - no global region in raw RegionManager
    assert_eq!(manager.regions.len(), 0);

    // Add new regions
    let id1 = manager.create_region();
    let id2 = manager.create_region();

    assert_ne!(id1, id2);

    // Get regions
    assert!(manager.get_region(id1).is_some());
    assert!(manager.get_region(id2).is_some());

    // Get mutable references
    assert!(manager.get_region_mut(id1).is_some());
    assert!(manager.get_region_mut(id2).is_some());
}

#[test]
fn test_value_with_defining_op() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    let result_name = ctx.intern_string("result");

    // Create a value and operation
    let val = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(result_name),
            ty: i32_type,
            defining_op: None,
        })
    };

    let const_op = ConstantOp {
        result: val,
        value: Attribute::Integer(100),
    };

    let global_region = ctx.global_region();
    let op_data = const_op.into_op_data(&mut ctx, global_region);

    // Add operation and update defining_op
    let op_ref = {
        let region = ctx.get_global_region_mut();
        let op_ref = region.add_op(op_data);

        // Update the value's defining op
        region.values.get_mut(val).unwrap().defining_op = Some(uvir::ops::OpRef(op_ref));
        op_ref
    };

    // Verify the relationship
    let region = ctx.get_global_region();
    let value = region.values.get(val).unwrap();
    match value.defining_op {
        Some(uvir::ops::OpRef(opr)) => assert_eq!(opr, op_ref),
        None => panic!("Expected defining op to be set"),
    }

    let op = region.get_op(op_ref).unwrap();
    assert_eq!(op.results[0], val);
}
