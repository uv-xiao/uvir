use uvir::dialects::affine_derive::*;
use uvir::dialects::builtin::integer_type;
use uvir::*;

#[test]
fn test_affine_for() {
    let mut ctx = Context::new();

    // Create loop bounds and step
    let index_ty = integer_type(&mut ctx, 64, false);
    let lower = ctx.create_value(Some("lower"), index_ty);
    let upper = ctx.create_value(Some("upper"), index_ty);
    let step = ctx.create_value(Some("step"), index_ty);
    let result = ctx.create_value(Some("result"), index_ty);
    let body = ctx.create_region();

    // Create affine maps as attributes
    let lower_map = Attribute::String(ctx.intern_string("(d0) -> (d0)"));
    let upper_map = Attribute::String(ctx.intern_string("(d0) -> (d0 + 10)"));

    let for_op = AffineForOp {
        lower_bound: lower,
        upper_bound: upper,
        step,
        results: result,
        body,
        lower_bound_map: lower_map,
        upper_bound_map: upper_map,
    };

    let global_region = ctx.global_region();
    let op_data = for_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "affine");
    assert_eq!(op_data.info.name, "for");
    assert_eq!(op_data.operands.len(), 3);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.regions.len(), 1);
    assert_eq!(op_data.attributes.len(), 2);
}

#[test]
fn test_affine_parallel() {
    let mut ctx = Context::new();

    // Create parallel loop bounds
    let index_ty = integer_type(&mut ctx, 64, false);
    let lower = ctx.create_value(Some("lower"), index_ty);
    let upper = ctx.create_value(Some("upper"), index_ty);
    let step = ctx.create_value(Some("step"), index_ty);
    let result = ctx.create_value(Some("result"), index_ty);
    let body = ctx.create_region();

    let reductions = Attribute::String(ctx.intern_string("add"));

    let parallel_op = AffineParallelOp {
        lower_bounds: lower,
        upper_bounds: upper,
        steps: step,
        results: result,
        body,
        reductions,
    };

    let global_region = ctx.global_region();
    let op_data = parallel_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "affine");
    assert_eq!(op_data.info.name, "parallel");
    assert_eq!(op_data.operands.len(), 3);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.regions.len(), 1);
}

#[test]
fn test_affine_if() {
    let mut ctx = Context::new();

    // Create operands and regions
    let index_ty = integer_type(&mut ctx, 64, false);
    let i32_ty = integer_type(&mut ctx, 32, true);
    let operand = ctx.create_value(Some("idx"), index_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    let then_region = ctx.create_region();
    let else_region = ctx.create_region();

    let condition_set = Attribute::String(ctx.intern_string("(d0) : (d0 >= 0)"));

    let if_op = AffineIfOp {
        operands: operand,
        results: result,
        then_region,
        else_region,
        condition_set,
    };

    let global_region = ctx.global_region();
    let op_data = if_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "affine");
    assert_eq!(op_data.info.name, "if");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.regions.len(), 2);
}

#[test]
fn test_affine_apply() {
    let mut ctx = Context::new();

    // Create index values
    let index_ty = integer_type(&mut ctx, 64, false);
    let i = ctx.create_value(Some("i"), index_ty);
    let result = ctx.create_value(Some("result"), index_ty);

    let map = Attribute::String(ctx.intern_string("(d0) -> (d0 * 2 + 1)"));

    let apply_op = AffineApplyOp {
        operands: i,
        result,
        map,
    };

    let global_region = ctx.global_region();
    let op_data = apply_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "affine");
    assert_eq!(op_data.info.name, "apply");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 1);
}

#[test]
fn test_affine_load_store() {
    let mut ctx = Context::new();

    // Create memref and indices
    let f32_ty = integer_type(&mut ctx, 32, true); // Using integer as placeholder
    let index_ty = integer_type(&mut ctx, 64, false);
    let memref_ty = integer_type(&mut ctx, 64, false); // Placeholder for memref type

    let memref = ctx.create_value(Some("mem"), memref_ty);
    let index = ctx.create_value(Some("idx"), index_ty);
    let value = ctx.create_value(Some("val"), f32_ty);
    let loaded = ctx.create_value(Some("loaded"), f32_ty);

    let map = Attribute::String(ctx.intern_string("(d0) -> (d0)"));

    // Test load
    let load_op = AffineLoadOp {
        memref: memref,
        indices: index,
        result: loaded,
        map: map.clone(),
    };

    let global_region = ctx.global_region();
    let load_data = load_op.into_op_data(&mut ctx, global_region);
    assert_eq!(load_data.info.dialect, "affine");
    assert_eq!(load_data.info.name, "load");

    // Test store
    let store_op = AffineStoreOp {
        value,
        memref,
        indices: index,
        map,
    };

    let store_data = store_op.into_op_data(&mut ctx, global_region);
    assert_eq!(store_data.info.dialect, "affine");
    assert_eq!(store_data.info.name, "store");
    assert_eq!(store_data.operands.len(), 3);
    assert_eq!(store_data.results.len(), 0);
}

#[test]
fn test_affine_min_max() {
    let mut ctx = Context::new();

    // Create operands
    let index_ty = integer_type(&mut ctx, 64, false);
    let i = ctx.create_value(Some("i"), index_ty);
    let min_result = ctx.create_value(Some("min"), index_ty);
    let max_result = ctx.create_value(Some("max"), index_ty);

    let map = Attribute::String(ctx.intern_string("(d0) -> (d0, 100)"));

    // Test min
    let min_op = AffineMinOp {
        operands: i,
        result: min_result,
        map: map.clone(),
    };

    let global_region = ctx.global_region();
    let min_data = min_op.into_op_data(&mut ctx, global_region);
    assert_eq!(min_data.info.dialect, "affine");
    assert_eq!(min_data.info.name, "min");

    // Test max
    let max_op = AffineMaxOp {
        operands: i,
        result: max_result,
        map,
    };

    let max_data = max_op.into_op_data(&mut ctx, global_region);
    assert_eq!(max_data.info.dialect, "affine");
    assert_eq!(max_data.info.name, "max");
}

#[test]
fn test_affine_yield() {
    let mut ctx = Context::new();

    // Create value to yield
    let i32_ty = integer_type(&mut ctx, 32, true);
    let value = ctx.create_value(Some("val"), i32_ty);

    let yield_op = AffineYieldOp { operands: value };

    let global_region = ctx.global_region();
    let op_data = yield_op.into_op_data(&mut ctx, global_region);

    assert_eq!(op_data.info.dialect, "affine");
    assert_eq!(op_data.info.name, "yield");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 0);
}

#[test]
fn test_roundtrip_affine_apply() {
    let mut ctx = Context::new();

    // Create apply operation
    let index_ty = integer_type(&mut ctx, 64, false);
    let operand = ctx.create_value(Some("idx"), index_ty);
    let result = ctx.create_value(Some("result"), index_ty);
    let map = Attribute::String(ctx.intern_string("(d0) -> (d0 + 1)"));

    let apply_op = AffineApplyOp {
        operands: operand,
        result,
        map: map.clone(),
    };

    let apply_clone = apply_op.clone();
    let global_region = ctx.global_region();
    let op_data = apply_op.into_op_data(&mut ctx, global_region);

    // Convert back
    let recovered = AffineApplyOp::from_op_data(&op_data, &ctx);

    // Check fields match
    assert_eq!(recovered.operands, apply_clone.operands);
    assert_eq!(recovered.result, apply_clone.result);
    // Note: Attribute comparison may not work directly due to the placeholder implementation
}
