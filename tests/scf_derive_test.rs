use uvir::*;
use uvir::dialects::builtin::integer_type;
use uvir::dialects::scf_derive::*;

#[test]
fn test_for_loop() {
    let mut ctx = Context::new();
    
    // Create loop bounds
    let index_ty = integer_type(&mut ctx, 64, false); // index type
    let lower = ctx.create_value(Some("lower"), index_ty);
    let upper = ctx.create_value(Some("upper"), index_ty);
    let step = ctx.create_value(Some("step"), index_ty);
    let result = ctx.create_value(Some("result"), index_ty);
    
    // Create loop body region
    let body = ctx.create_region();
    
    let for_op = ForOp {
        lower_bound: lower,
        upper_bound: upper,
        step,
        results: result,
        body,
    };
    
    let op_data = for_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "scf");
    assert_eq!(op_data.info.name, "for");
    assert_eq!(op_data.operands.len(), 3);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.regions.len(), 1);
}

#[test]
fn test_if_op() {
    let mut ctx = Context::new();
    
    // Create condition and result values
    let bool_ty = integer_type(&mut ctx, 1, false);
    let i32_ty = integer_type(&mut ctx, 32, true);
    let condition = ctx.create_value(Some("cond"), bool_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    
    // Create then and else regions
    let then_region = ctx.create_region();
    let else_region = ctx.create_region();
    
    let if_op = IfOp {
        condition,
        results: result,
        then_region,
        else_region,
    };
    
    let op_data = if_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "scf");
    assert_eq!(op_data.info.name, "if");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.regions.len(), 2);
}

#[test]
fn test_while_loop() {
    let mut ctx = Context::new();
    
    // Create initial value and result
    let i32_ty = integer_type(&mut ctx, 32, true);
    let init = ctx.create_value(Some("init"), i32_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    
    // Create condition and body regions
    let before = ctx.create_region();
    let after = ctx.create_region();
    
    let while_op = WhileOp {
        init_args: init,
        results: result,
        before_region: before,
        after_region: after,
    };
    
    let op_data = while_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "scf");
    assert_eq!(op_data.info.name, "while");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.regions.len(), 2);
}

#[test]
fn test_yield_op() {
    let mut ctx = Context::new();
    
    // Create value to yield
    let i32_ty = integer_type(&mut ctx, 32, true);
    let value = ctx.create_value(Some("value"), i32_ty);
    
    let yield_op = YieldOp {
        operands: value,
    };
    
    let op_data = yield_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "scf");
    assert_eq!(op_data.info.name, "yield");
    assert_eq!(op_data.operands.len(), 1);
    assert_eq!(op_data.results.len(), 0);
}

#[test]
fn test_condition_op() {
    let mut ctx = Context::new();
    
    // Create condition and argument values
    let bool_ty = integer_type(&mut ctx, 1, false);
    let i32_ty = integer_type(&mut ctx, 32, true);
    let cond = ctx.create_value(Some("cond"), bool_ty);
    let arg = ctx.create_value(Some("arg"), i32_ty);
    
    let condition_op = ConditionOp {
        condition: cond,
        args: arg,
    };
    
    let op_data = condition_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "scf");
    assert_eq!(op_data.info.name, "condition");
    assert_eq!(op_data.operands.len(), 2);
    assert_eq!(op_data.results.len(), 0);
}

#[test]
fn test_execute_region() {
    let mut ctx = Context::new();
    
    // Create result value and region
    let i32_ty = integer_type(&mut ctx, 32, true);
    let result = ctx.create_value(Some("result"), i32_ty);
    let body = ctx.create_region();
    
    let exec_op = ExecuteRegionOp {
        results: result,
        body,
    };
    
    let op_data = exec_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "scf");
    assert_eq!(op_data.info.name, "execute_region");
    assert_eq!(op_data.operands.len(), 0);
    assert_eq!(op_data.results.len(), 1);
    assert_eq!(op_data.regions.len(), 1);
}

#[test]
fn test_parallel_op() {
    let mut ctx = Context::new();
    
    // Create bounds for parallel loop
    let index_ty = integer_type(&mut ctx, 64, false);
    let lower = ctx.create_value(Some("lower"), index_ty);
    let upper = ctx.create_value(Some("upper"), index_ty);
    let step = ctx.create_value(Some("step"), index_ty);
    let body = ctx.create_region();
    
    let parallel_op = ParallelOp {
        lower_bounds: lower,
        upper_bounds: upper,
        steps: step,
        body,
    };
    
    let op_data = parallel_op.into_op_data(&mut ctx);
    
    assert_eq!(op_data.info.dialect, "scf");
    assert_eq!(op_data.info.name, "parallel");
    assert_eq!(op_data.operands.len(), 3);
    assert_eq!(op_data.results.len(), 0);
    assert_eq!(op_data.regions.len(), 1);
}

#[test]
fn test_roundtrip_if_op() {
    let mut ctx = Context::new();
    
    // Create if operation
    let bool_ty = integer_type(&mut ctx, 1, false);
    let i32_ty = integer_type(&mut ctx, 32, true);
    let condition = ctx.create_value(Some("cond"), bool_ty);
    let result = ctx.create_value(Some("result"), i32_ty);
    let then_region = ctx.create_region();
    let else_region = ctx.create_region();
    
    let if_op = IfOp {
        condition,
        results: result,
        then_region,
        else_region,
    };
    
    let if_clone = if_op.clone();
    let op_data = if_op.into_op_data(&mut ctx);
    
    // Convert back
    let recovered = IfOp::from_op_data(&op_data, &ctx);
    
    // Check fields match
    assert_eq!(recovered.condition, if_clone.condition);
    assert_eq!(recovered.results, if_clone.results);
    assert_eq!(recovered.then_region, if_clone.then_region);
    assert_eq!(recovered.else_region, if_clone.else_region);
}