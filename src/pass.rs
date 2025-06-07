use crate::context::Context;
use crate::ops::{OpData, Val, Opr};
use crate::region::RegionId;
use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use ahash::AHashSet;

// Pattern-based rewriting trait
pub trait RewritePattern: 'static {
    fn benefit(&self) -> usize { 1 } // Priority
    
    fn match_and_rewrite(
        &self,
        op: Opr,
        rewriter: &mut PatternRewriter,
    ) -> Result<bool>;
}

// Rewriter with operation tracking
pub struct PatternRewriter<'a> {
    pub ctx: &'a mut Context,
    worklist: VecDeque<Opr>,
    erased: AHashSet<Opr>,
    current_region: RegionId,
}

impl<'a> PatternRewriter<'a> {
    pub fn new(ctx: &'a mut Context, region: RegionId) -> Self {
        let mut worklist = VecDeque::new();
        
        // Add all operations in the region to the worklist
        if let Some(region_ref) = ctx.get_region(region) {
            for (opr, _) in region_ref.iter_ops() {
                worklist.push_back(opr);
            }
        }
        
        Self {
            ctx,
            worklist,
            erased: AHashSet::new(),
            current_region: region,
        }
    }

    // Replace an operation with new operations
    pub fn replace_op(&mut self, op: Opr, new_ops: &[Opr]) {
        self.erased.insert(op);
        
        // Add new operations to worklist
        for &new_op in new_ops {
            self.worklist.push_back(new_op);
        }
    }

    // Erase an operation
    pub fn erase_op(&mut self, op: Opr) {
        self.erased.insert(op);
        
        if let Some(region) = self.ctx.get_region_mut(self.current_region) {
            region.remove_op(op);
        }
    }

    // Replace all uses of a value with another value
    pub fn replace_all_uses(&mut self, from: Val, to: Val) {
        if let Some(region) = self.ctx.get_region_mut(self.current_region) {
            // Update all operations that use 'from' to use 'to' instead
            let ops_to_update: Vec<Opr> = region.op_order.clone();
            
            for opr in ops_to_update {
                if let Some(op) = region.get_op_mut(opr) {
                    for operand in &mut op.operands {
                        if *operand == from {
                            *operand = to;
                        }
                    }
                }
            }
        }
    }

    // Get the next operation from the worklist
    pub fn next_op(&mut self) -> Option<Opr> {
        while let Some(op) = self.worklist.pop_front() {
            if !self.erased.contains(&op) {
                return Some(op);
            }
        }
        None
    }

    // Create a new operation and add it to the region
    pub fn create_op(&mut self, op_data: OpData) -> Opr {
        if let Some(region) = self.ctx.get_region_mut(self.current_region) {
            let opr = region.add_operation(op_data);
            self.worklist.push_back(opr);
            opr
        } else {
            panic!("Invalid region");
        }
    }
}

// Greedy pattern driver
pub fn apply_patterns_greedy(
    ctx: &mut Context,
    patterns: &[Box<dyn RewritePattern>],
    region: RegionId,
) -> Result<bool> {
    let mut changed = false;
    
    // Sort patterns by benefit (using indices to avoid cloning)
    let mut pattern_indices: Vec<usize> = (0..patterns.len()).collect();
    pattern_indices.sort_by_key(|&i| std::cmp::Reverse(patterns[i].benefit()));
    
    // Fixed-point iteration
    loop {
        let mut local_changed = false;
        let mut rewriter = PatternRewriter::new(ctx, region);
        
        while let Some(op) = rewriter.next_op() {
            // Skip if operation was erased
            if rewriter.erased.contains(&op) {
                continue;
            }
            
            // Try each pattern (in sorted order)
            for &idx in &pattern_indices {
                if patterns[idx].match_and_rewrite(op, &mut rewriter)? {
                    local_changed = true;
                    break;
                }
            }
        }
        
        if !local_changed {
            break;
        }
        changed = true;
    }
    
    Ok(changed)
}

// Analysis-based passes
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&mut self, ctx: &mut Context) -> Result<PassResult>;
}

#[derive(Default)]
pub struct PassResult {
    pub changed: bool,
    pub statistics: HashMap<String, u64>,
}

impl PassResult {
    pub fn new() -> Self {
        Self {
            changed: false,
            statistics: HashMap::new(),
        }
    }

    pub fn with_change(mut self) -> Self {
        self.changed = true;
        self
    }

    pub fn add_statistic(&mut self, name: &str, value: u64) {
        self.statistics.insert(name.to_string(), value);
    }
}

// Pass manager with dependency resolution
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
    dependencies: HashMap<String, Vec<String>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            dependencies: HashMap::new(),
        }
    }

    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn add_dependency(&mut self, pass_name: &str, depends_on: &str) {
        self.dependencies
            .entry(pass_name.to_string())
            .or_insert_with(Vec::new)
            .push(depends_on.to_string());
    }

    pub fn run(&mut self, ctx: &mut Context) -> Result<()> {
        // For now, run passes in order
        // TODO: Implement proper dependency resolution
        for pass in &mut self.passes {
            let name = pass.name().to_string();
            println!("Running pass: {}", name);
            
            let result = pass.run(ctx)?;
            
            if result.changed {
                println!("  Pass {} made changes", name);
            }
            
            for (stat_name, value) in &result.statistics {
                println!("  {}: {}", stat_name, value);
            }
        }
        
        Ok(())
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

// Example pattern: constant folding for add
pub struct ConstantFoldAddPattern;

impl RewritePattern for ConstantFoldAddPattern {
    fn benefit(&self) -> usize {
        10 // High priority
    }

    fn match_and_rewrite(
        &self,
        op: Opr,
        rewriter: &mut PatternRewriter,
    ) -> Result<bool> {
        // Get the operation
        let region = rewriter.ctx.get_region(rewriter.current_region)
            .ok_or_else(|| crate::error::Error::InvalidRegion("Region not found".to_string()))?;
        
        let op_data = region.get_op(op)
            .ok_or_else(|| crate::error::Error::InvalidOperation("Operation not found".to_string()))?;
        
        // Check if it's an add operation
        if op_data.info.dialect != "arith" || op_data.info.name != "addi" {
            return Ok(false);
        }
        
        // Check if both operands are constants
        // TODO: Implement constant checking logic
        
        Ok(false)
    }
}

// Example pass: dead code elimination
pub struct DeadCodeEliminationPass;

impl Pass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }

    fn run(&mut self, ctx: &mut Context) -> Result<PassResult> {
        let mut result = PassResult::new();
        let mut removed_count = 0;
        
        // For each region
        let global_region = ctx.global_region();
        if let Some(region) = ctx.get_region_mut(global_region) {
            // Build use-def chains
            let mut value_uses: HashMap<Val, Vec<Opr>> = HashMap::new();
            
            for (opr, op) in region.iter_ops() {
                for &operand in &op.operands {
                    value_uses.entry(operand).or_insert_with(Vec::new).push(opr);
                }
            }
            
            // Find dead operations (operations whose results are never used)
            let ops_to_remove: Vec<Opr> = region.op_order.iter()
                .filter(|&&opr| {
                    if let Some(op) = region.get_op(opr) {
                        // Check if any result is used
                        !op.results.iter().any(|&result| {
                            value_uses.contains_key(&result)
                        })
                    } else {
                        false
                    }
                })
                .copied()
                .collect();
            
            // Remove dead operations
            for opr in ops_to_remove {
                region.remove_op(opr);
                removed_count += 1;
                result.changed = true;
            }
        }
        
        result.add_statistic("operations_removed", removed_count);
        Ok(result)
    }
}