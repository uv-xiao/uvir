// Tests for the pass and rewrite infrastructure in uvir.
//
// Purpose: Validates the pass system functionality including:
// - Pattern-based rewriting with the RewritePattern trait
// - Constant folding optimization patterns
// - Dead code elimination passes
// - Pattern application with greedy driver
// - Pass manager for organizing optimization pipelines
// - Pattern benefit/priority system
//
// The pass infrastructure enables IR transformations and optimizations,
// essential for improving code quality and enabling domain-specific optimizations.
//
// Note: Since the pass module is not exposed in the public API,
// these tests are commented out but demonstrate the intended functionality.

use uvir::attribute::Attribute;
use uvir::dialects::arith::{ConstantOp, AddOp};
use uvir::dialects::builtin::integer_type;
use uvir::{Context, Printer, Value};
use uvir::pass::{RewritePattern, PatternRewriter, apply_patterns_greedy, apply_patterns_greedy_all_regions, Pass};
use uvir::Opr;

// Example test demonstrating how constant folding would work
#[test]
fn test_constant_folding_concept() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create two constants
    let c1_name = ctx.intern_string("c1");
    let c2_name = ctx.intern_string("c2");
    let (c1_val, c2_val) = {
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
        (c1, c2)
    };

    // Create constant operations
    let const1 = ConstantOp {
        result: c1_val,
        value: Attribute::Integer(10),
    };
    let const2 = ConstantOp {
        result: c2_val,
        value: Attribute::Integer(20),
    };

    // In a real implementation, constant folding would:
    // 1. Detect that both operands to an add are constants
    // 2. Compute the result at compile time (10 + 20 = 30)
    // 3. Replace the add operation with a constant operation

    let global_region = ctx.global_region();
    let const1_data = const1.into_op_data(&mut ctx, global_region);
    let const2_data = const2.into_op_data(&mut ctx, global_region);

    {
        let region = ctx.get_global_region_mut();
        region.add_op(const1_data);
        region.add_op(const2_data);
    }

    // This test demonstrates the concept without the actual pass implementation
    assert_eq!(ctx.get_global_region().operations.len(), 2);
}

#[test]
fn test_constant_fold_add_pattern() {
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create the pattern
    let pattern = ConstantFoldAddPattern;

    // Create constants and add operation: 10 + 20
    let c1_name = ctx.intern_string("c1");
    let c2_name = ctx.intern_string("c2");
    let sum_name = ctx.intern_string("sum");
    
    let (c1_val, c2_val, sum_val) = {
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

        let sum = region.add_value(Value {
            name: Some(sum_name),
            ty: i32_type,
            defining_op: None,
        });

        (c1, c2, sum)
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

    let add = AddOp {
        result: sum_val,
        lhs: c1_val,
        rhs: c2_val,
    };

    // Add operations to region
    let global_region = ctx.global_region();
    let op_data1 = const1.into_op_data(&mut ctx, global_region);
    let op_data2 = const2.into_op_data(&mut ctx, global_region);
    let add_data = add.into_op_data(&mut ctx, global_region);

    {
        let region = ctx.get_global_region_mut();
        let op1 = region.add_op(op_data1);
        let op2 = region.add_op(op_data2);
        let _op3 = region.add_op(add_data);
        
        // Set defining ops
        if let Some(v) = region.get_value_mut(c1_val) {
            v.defining_op = Some(uvir::OpRef(op1));
        }
        if let Some(v) = region.get_value_mut(c2_val) {
            v.defining_op = Some(uvir::OpRef(op2));
        }
    }

    // Apply the pattern using apply_patterns_greedy
    let patterns: Vec<Box<dyn RewritePattern>> = vec![Box::new(pattern)];

    // print before applying patterns
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    println!("Before applying patterns: {}", printer.get_output());

    let global_region = ctx.global_region();
    let changed = apply_patterns_greedy(&mut ctx, &patterns, global_region).unwrap();

    // print after applying patterns
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    println!("After applying patterns: {}", printer.get_output());

    // The pattern should have matched and folded the constants
    assert!(changed, "Pattern should have matched and folded the add");

    // Verify that we now have 3 operations (2 original constants + 1 new folded constant)
    // The add operation should have been replaced
    let value_key = ctx.intern_string("value");
    let region = ctx.get_global_region();
    assert_eq!(region.operations.len(), 3, "Should have 3 operations after folding");
    
    // Check that the result is a constant with value 30
    let has_folded_constant = region.iter_ops().any(|(_, op)| {
        op.info.name == "constant" && 
        {
            use uvir::AttributeMapExt;
            op.attributes.get(value_key) == Some(&Attribute::Integer(30))
        }
    });
    assert!(has_folded_constant, "Should have a constant with value 30");

    // Use dead code elimination pass
    let mut dce_pass = uvir::pass::DeadCodeEliminationPass;
    let _ = dce_pass.run(&mut ctx).unwrap();

    // print after running DCE
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    println!("After DCE: {}", printer.get_output());
}

#[test]
fn test_pattern_benefit() {
    // Test that patterns are sorted by benefit
    let patterns: Vec<Box<dyn RewritePattern>> = vec![
        Box::new(LowBenefitPattern),
        Box::new(HighBenefitPattern),
        Box::new(MediumBenefitPattern),
    ];

    let mut sorted = patterns;
    sorted.sort_by_key(|p| std::cmp::Reverse(p.benefit()));

    // High benefit should come first
    assert_eq!(sorted[0].benefit(), 10);
    assert_eq!(sorted[1].benefit(), 5);
    assert_eq!(sorted[2].benefit(), 1);
}

// Test patterns with different benefits
struct HighBenefitPattern;
struct MediumBenefitPattern;
struct LowBenefitPattern;
struct ConstantFoldAddPattern;

impl RewritePattern for HighBenefitPattern {
    fn benefit(&self) -> usize { 10 }

    fn match_and_rewrite(
        &self,
        _op: Opr,
        _rewriter: &mut PatternRewriter,
    ) -> uvir::Result<bool> {
        Ok(false)
    }
}

impl RewritePattern for MediumBenefitPattern {
    fn benefit(&self) -> usize { 5 }

    fn match_and_rewrite(
        &self,
        _op: Opr,
        _rewriter: &mut PatternRewriter,
    ) -> uvir::Result<bool> {
        Ok(false)
    }
}

impl RewritePattern for LowBenefitPattern {
    fn benefit(&self) -> usize { 1 }

    fn match_and_rewrite(
        &self,
        _op: Opr,
        _rewriter: &mut PatternRewriter,
    ) -> uvir::Result<bool> {
        Ok(false)
    }
}

impl RewritePattern for ConstantFoldAddPattern {
    fn benefit(&self) -> usize {
        10 // High priority
    }

    fn match_and_rewrite(&self, op: Opr, rewriter: &mut PatternRewriter) -> uvir::Result<bool> {
        // Import the trait
        use uvir::AttributeMapExt;
        
        // Intern the value key first
        let value_key = rewriter.ctx.intern_string("value");
        
        // Get the operation from the current region being processed
        let current_region = rewriter.current_region();
        let region = rewriter.ctx.get_region(current_region).ok_or_else(|| {
            uvir::Error::InvalidRegion("Current region not found".to_string())
        })?;

        let op_data = region.get_op(op).ok_or_else(|| {
            uvir::Error::InvalidOperation("Operation not found".to_string())
        })?;

        // Check if it's an add operation
        if op_data.info.dialect != "arith" || op_data.info.name != "addi" {
            return Ok(false);
        }

        // Get operands
        if op_data.operands.len() != 2 || op_data.results.len() != 1 {
            return Ok(false);
        }

        let lhs_ref = op_data.operands[0];
        let rhs_ref = op_data.operands[1];
        let result = op_data.results[0];

        // Extract the Val from ValueRef
        let lhs = lhs_ref.val;
        let rhs = rhs_ref.val;

        // Find the defining operations for the operands
        let lhs_def = region.get_value(lhs).and_then(|v| v.defining_op);
        let rhs_def = region.get_value(rhs).and_then(|v| v.defining_op);

        if lhs_def.is_none() || rhs_def.is_none() {
            return Ok(false);
        }

        // Check if both are constants
        let lhs_op = region.get_op(lhs_def.unwrap().0);
        let rhs_op = region.get_op(rhs_def.unwrap().0);

        if lhs_op.is_none() || rhs_op.is_none() {
            return Ok(false);
        }

        let lhs_op = lhs_op.unwrap();
        let rhs_op = rhs_op.unwrap();

        if lhs_op.info.dialect != "arith" || lhs_op.info.name != "constant" ||
           rhs_op.info.dialect != "arith" || rhs_op.info.name != "constant" {
            return Ok(false);
        }
        
        let lhs_value = lhs_op.attributes.get(value_key)
            .and_then(|attr| match attr {
                Attribute::Integer(v) => Some(*v),
                _ => None,
            });
            
        let rhs_value = rhs_op.attributes.get(value_key)
            .and_then(|attr| match attr {
                Attribute::Integer(v) => Some(*v),
                _ => None,
            });

        if lhs_value.is_none() || rhs_value.is_none() {
            return Ok(false);
        }

        // Compute the folded value
        let folded_value = lhs_value.unwrap() + rhs_value.unwrap();

        // Create a new constant operation with the folded value
        let folded_const = ConstantOp {
            result,
            value: Attribute::Integer(folded_value),
        };

        let new_op_data = folded_const.into_op_data(rewriter.ctx, current_region);
        let new_op = rewriter.create_op(new_op_data);

        
        // Erase the original add operation
        rewriter.erase_op(op);
        
        // Add the new constant to replace the add
        rewriter.replace_op(op, &[new_op]);

        Ok(true)
    }
}

#[test]
fn test_dead_code_elimination() {
    
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create an unused value
    let unused_name = ctx.intern_string("unused");
    let unused_val = {
        let region = ctx.get_global_region_mut();
        region.add_value(Value {
            name: Some(unused_name),
            ty: i32_type,
            defining_op: None,
        })
    };

    // Create an unused constant operation
    let unused_const = ConstantOp {
        result: unused_val,
        value: Attribute::Integer(42),
    };

    let global_region = ctx.global_region();
    let op_data = unused_const.into_op_data(&mut ctx, global_region);
    {
        let region = ctx.get_global_region_mut();
        region.add_op(op_data);
    }

    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    println!("Before DCE: {}", printer.get_output());

    // Create and run the dead code elimination pass from pass.rs
    let mut dce_pass = uvir::pass::DeadCodeEliminationPass;
    let result = dce_pass.run(&mut ctx).unwrap();

    // print after running DCE
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    println!("After DCE: {}", printer.get_output());  

    assert!(result.changed, "DCE should have made changes");
    assert_eq!(*result.statistics.get("operations_removed").unwrap_or(&0), 1);
    assert_eq!(ctx.get_global_region().operations.len(), 0, "Unused operation should be removed");
    println!("DCE result: {:?}", result.statistics);
}

#[test]
fn test_pass_manager() {
    use uvir::pass::PassManager;
    
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);

    // Create some unused operations
    for i in 0..3 {
        let name = ctx.intern_string(&format!("unused{}", i));
        let val = {
            let region = ctx.get_global_region_mut();
            region.add_value(Value {
                name: Some(name),
                ty: i32_type,
                defining_op: None,
            })
        };

        let const_op = ConstantOp {
            result: val,
            value: Attribute::Integer(i as i64),
        };

        let global_region = ctx.global_region();
        let op_data = const_op.into_op_data(&mut ctx, global_region);
        ctx.get_global_region_mut().add_op(op_data);
    }

    // Create a pass manager and add passes
    let mut pass_manager = PassManager::new();
    pass_manager.add_pass(Box::new(uvir::pass::DeadCodeEliminationPass));

    // Run the pass manager
    pass_manager.run(&mut ctx).unwrap();

    // All unused operations should be removed
    assert_eq!(ctx.get_global_region().operations.len(), 0, "All unused operations should be removed");
}

#[test]
fn test_multi_region_pattern_rewriting() {
    use uvir::dialects::scf_derive::{ForOp, YieldOp};
    use uvir::dialects::builtin::integer_type;
    
    let mut ctx = Context::new();
    let i32_type = integer_type(&mut ctx, 32, true);
    
    // Create a simple for loop with a nested region
    let loop_body = ctx.create_region_with_parent(ctx.global_region());
    
    // Create loop bounds in the global region
    let lower_name = ctx.intern_string("lower");
    let upper_name = ctx.intern_string("upper");
    let step_name = ctx.intern_string("step");
    let result_name = ctx.intern_string("result");
    
    let (lower_val, upper_val, step_val, result_val) = {
        let region = ctx.get_global_region_mut();
        let lower = region.add_value(Value {
            name: Some(lower_name),
            ty: i32_type,
            defining_op: None,
        });
        let upper = region.add_value(Value {
            name: Some(upper_name),
            ty: i32_type,
            defining_op: None,
        });
        let step = region.add_value(Value {
            name: Some(step_name),
            ty: i32_type,
            defining_op: None,
        });
        let result = region.add_value(Value {
            name: Some(result_name),
            ty: i32_type,
            defining_op: None,
        });
        (lower, upper, step, result)
    };
    
    // Create constants for loop bounds
    let lower_const = ConstantOp {
        result: lower_val,
        value: Attribute::Integer(0),
    };
    let upper_const = ConstantOp {
        result: upper_val,
        value: Attribute::Integer(10),
    };
    let step_const = ConstantOp {
        result: step_val,
        value: Attribute::Integer(1),
    };
    
    let global_region = ctx.global_region();
    let lower_data = lower_const.into_op_data(&mut ctx, global_region);
    let upper_data = upper_const.into_op_data(&mut ctx, global_region);
    let step_data = step_const.into_op_data(&mut ctx, global_region);
    
    {
        let region = ctx.get_global_region_mut();
        let op1 = region.add_op(lower_data);
        let op2 = region.add_op(upper_data);
        let op3 = region.add_op(step_data);
        
        // Set defining ops
        if let Some(v) = region.get_value_mut(lower_val) {
            v.defining_op = Some(uvir::OpRef(op1));
        }
        if let Some(v) = region.get_value_mut(upper_val) {
            v.defining_op = Some(uvir::OpRef(op2));
        }
        if let Some(v) = region.get_value_mut(step_val) {
            v.defining_op = Some(uvir::OpRef(op3));
        }
    }
    
    // In the loop body: constant folding opportunity
    let c1_name = ctx.intern_string("c1");
    let c2_name = ctx.intern_string("c2");
    let sum_name = ctx.intern_string("sum");
    
    let (c1_val, c2_val, sum_val) = {
        let region = ctx.get_region_mut(loop_body).unwrap();
        
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
        
        let sum = region.add_value(Value {
            name: Some(sum_name),
            ty: i32_type,
            defining_op: None,
        });
        
        (c1, c2, sum)
    };
    
    // Create operations in loop body
    let const1 = ConstantOp {
        result: c1_val,
        value: Attribute::Integer(15),
    };
    
    let const2 = ConstantOp {
        result: c2_val,
        value: Attribute::Integer(25),
    };
    
    let add = AddOp {
        result: sum_val,
        lhs: c1_val,
        rhs: c2_val,
    };
    
    let yield_op = YieldOp {
        operands: sum_val,
    };
    
    let const1_data = const1.into_op_data(&mut ctx, loop_body);
    let const2_data = const2.into_op_data(&mut ctx, loop_body);
    let add_data = add.into_op_data(&mut ctx, loop_body);
    let yield_data = yield_op.into_op_data(&mut ctx, loop_body);
    
    {
        let region = ctx.get_region_mut(loop_body).unwrap();
        let op1 = region.add_op(const1_data);
        let op2 = region.add_op(const2_data);
        let op3 = region.add_op(add_data);
        region.add_op(yield_data);
        
        // Set defining ops
        if let Some(v) = region.get_value_mut(c1_val) {
            v.defining_op = Some(uvir::OpRef(op1));
        }
        if let Some(v) = region.get_value_mut(c2_val) {
            v.defining_op = Some(uvir::OpRef(op2));
        }
        if let Some(v) = region.get_value_mut(sum_val) {
            v.defining_op = Some(uvir::OpRef(op3));
        }
    }
    
    // Create the for loop
    let for_op = ForOp {
        lower_bound: lower_val,
        upper_bound: upper_val,
        step: step_val,
        results: result_val,
        body: loop_body,
    };
    
    let global_region_final = ctx.global_region();
    let for_data = for_op.into_op_data(&mut ctx, global_region_final);
    ctx.get_global_region_mut().add_op(for_data);
    
    // Print before applying patterns
    println!("Before multi-region pattern application:");
    println!("Global region has {} operations", ctx.get_global_region().operations.len());
    println!("Loop body has {} operations", ctx.get_region(loop_body).unwrap().operations.len());
    
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    println!("Before applying patterns: {}", printer.get_output());

    // Apply constant folding pattern to all regions
    let patterns: Vec<Box<dyn RewritePattern>> = vec![Box::new(ConstantFoldAddPattern)];
    let global_region = ctx.global_region();
    let changed = apply_patterns_greedy_all_regions(&mut ctx, &patterns, global_region).unwrap();
    
    // print after applying patterns
    let mut printer = Printer::new();
    printer.print_region(&ctx, ctx.global_region()).unwrap();
    println!("After applying patterns: {}", printer.get_output());

    assert!(changed, "Pattern should have matched in nested regions");
    
    // Print after applying patterns
    println!("\nAfter multi-region pattern application:");
    println!("Global region has {} operations", ctx.get_global_region().operations.len());
    println!("Loop body has {} operations", ctx.get_region(loop_body).unwrap().operations.len());
    
    // Verify constants were folded in nested regions
    use uvir::AttributeMapExt;
    let value_key = ctx.intern_string("value");
    
    // Check loop body has folded constant (40 = 15 + 25)
    let loop_ref = ctx.get_region(loop_body).unwrap();
    let has_folded = loop_ref.iter_ops().any(|(_, op)| {
        op.info.name == "constant" && 
        op.attributes.get(value_key) == Some(&Attribute::Integer(40))
    });
    assert!(has_folded, "Loop body should have constant with value 40");
}
