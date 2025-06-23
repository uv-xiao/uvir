use crate::context::Context;
use crate::error::Result;
use crate::ops::{OpData, Opr, Val};
use crate::region::RegionId;
use ahash::AHashSet;
use std::collections::{HashMap, VecDeque};

// Pattern-based rewriting trait
pub trait RewritePattern: 'static {
    fn benefit(&self) -> usize {
        1
    } // Priority

    fn match_and_rewrite(&self, op: Opr, rewriter: &mut PatternRewriter) -> Result<bool>;
}

// Rewriter with operation tracking across multiple regions
pub struct PatternRewriter<'a> {
    pub ctx: &'a mut Context,
    worklist: VecDeque<(RegionId, Opr)>,
    erased: AHashSet<(RegionId, Opr)>,
    current_region: RegionId,
}

impl<'a> PatternRewriter<'a> {
    pub fn new(ctx: &'a mut Context, region: RegionId) -> Self {
        let mut worklist = VecDeque::new();

        // Add all operations in the region to the worklist
        if let Some(region_ref) = ctx.get_region(region) {
            for (opr, _) in region_ref.iter_ops() {
                worklist.push_back((region, opr));
            }
        }

        Self {
            ctx,
            worklist,
            erased: AHashSet::new(),
            current_region: region,
        }
    }

    // Get the current region being processed
    pub fn current_region(&self) -> RegionId {
        self.current_region
    }
    
    // Replace an operation with new operations in the current region
    pub fn replace_op(&mut self, op: Opr, new_ops: &[Opr]) {
        self.erased.insert((self.current_region, op));

        // Add new operations to worklist
        for &new_op in new_ops {
            self.worklist.push_back((self.current_region, new_op));
        }
    }
    
    // Replace an operation with new operations in a specific region
    pub fn replace_op_in_region(&mut self, region: RegionId, op: Opr, new_ops: &[Opr]) {
        self.erased.insert((region, op));

        // Add new operations to worklist
        for &new_op in new_ops {
            self.worklist.push_back((region, new_op));
        }
    }

    // Erase an operation from the current region
    pub fn erase_op(&mut self, op: Opr) {
        self.erased.insert((self.current_region, op));

        if let Some(region) = self.ctx.get_region_mut(self.current_region) {
            region.remove_op(op);
        }
    }
    
    // Erase an operation from a specific region
    pub fn erase_op_in_region(&mut self, region_id: RegionId, op: Opr) {
        self.erased.insert((region_id, op));

        if let Some(region) = self.ctx.get_region_mut(region_id) {
            region.remove_op(op);
        }
    }

    // Replace all uses of a value with another value in the current region
    pub fn replace_all_uses(&mut self, from: Val, to: Val) {
        self.replace_all_uses_in_region(self.current_region, from, to);
    }
    
    // Replace all uses of a value with another value in a specific region
    pub fn replace_all_uses_in_region(&mut self, region_id: RegionId, from: Val, to: Val) {
        if let Some(region) = self.ctx.get_region_mut(region_id) {
            // Update all operations that use 'from' to use 'to' instead
            let ops_to_update: Vec<Opr> = region.op_order.clone();

            for opr in ops_to_update {
                if let Some(op) = region.get_op_mut(opr) {
                    for operand in &mut op.operands {
                        // Check if this operand references the value we're replacing
                        if operand.val == from && operand.region == region_id {
                            // Update to the new value in the same region
                            operand.val = to;
                        }
                    }
                }
            }
        }
        
        // Also check nested regions
        self.replace_all_uses_in_nested_regions(region_id, from, to);
    }
    
    // Helper to replace uses in nested regions
    fn replace_all_uses_in_nested_regions(&mut self, parent_region: RegionId, from: Val, to: Val) {
        // Collect nested regions to avoid borrow checker issues
        let nested_regions: Vec<RegionId> = if let Some(region) = self.ctx.get_region(parent_region) {
            region.iter_ops()
                .flat_map(|(_, op)| op.regions.iter().copied())
                .collect()
        } else {
            vec![]
        };
        
        // Process nested regions
        for nested_region in nested_regions {
            self.replace_all_uses_in_region(nested_region, from, to);
        }
    }

    // Get the next operation from the worklist
    pub fn next_op(&mut self) -> Option<(RegionId, Opr)> {
        while let Some((region, op)) = self.worklist.pop_front() {
            if !self.erased.contains(&(region, op)) {
                self.current_region = region;
                return Some((region, op));
            }
        }
        None
    }

    // Create a new operation and add it to the current region
    pub fn create_op(&mut self, op_data: OpData) -> Opr {
        self.create_op_in_region(self.current_region, op_data)
    }
    
    // Create a new operation and add it to a specific region
    pub fn create_op_in_region(&mut self, region_id: RegionId, op_data: OpData) -> Opr {
        if let Some(region) = self.ctx.get_region_mut(region_id) {
            let opr = region.add_operation(op_data);
            self.worklist.push_back((region_id, opr));
            opr
        } else {
            panic!("Invalid region");
        }
    }
    
    // Add regions from an operation to the worklist for processing
    pub fn process_nested_regions(&mut self, op: &OpData) {
        for &region_id in &op.regions {
            if let Some(region) = self.ctx.get_region(region_id) {
                for (opr, _) in region.iter_ops() {
                    self.worklist.push_back((region_id, opr));
                }
            }
        }
    }
    
    // Get operation from a specific region
    pub fn get_op(&self, region_id: RegionId, op: Opr) -> Option<&OpData> {
        self.ctx.get_region(region_id)?.get_op(op)
    }
    
    // Get mutable operation from a specific region
    pub fn get_op_mut(&mut self, region_id: RegionId, op: Opr) -> Option<&mut OpData> {
        self.ctx.get_region_mut(region_id)?.get_op_mut(op)
    }
}

// Multi-region pattern driver that processes all regions recursively
pub fn apply_patterns_greedy_all_regions(
    ctx: &mut Context,
    patterns: &[Box<dyn RewritePattern>],
    start_region: RegionId,
) -> Result<bool> {
    let mut changed = false;
    let mut regions_to_process = vec![start_region];
    let mut processed_regions = AHashSet::new();
    
    // Collect all regions to process
    while let Some(region_id) = regions_to_process.pop() {
        if processed_regions.contains(&region_id) {
            continue;
        }
        processed_regions.insert(region_id);
        
        // Find nested regions
        if let Some(region) = ctx.get_region(region_id) {
            for (_, op) in region.iter_ops() {
                for &nested_region in &op.regions {
                    if !processed_regions.contains(&nested_region) {
                        regions_to_process.push(nested_region);
                    }
                }
            }
        }
    }
    
    // Apply patterns to each region
    for &region_id in &processed_regions {
        if apply_patterns_greedy(ctx, patterns, region_id)? {
            changed = true;
        }
    }
    
    Ok(changed)
}

// Greedy pattern driver for a single region
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

        while let Some((region, op)) = rewriter.next_op() {
            // Skip if operation was erased
            if rewriter.erased.contains(&(region, op)) {
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
                for operand_ref in &op.operands {
                    // Extract the Val from ValueRef for use tracking
                    value_uses.entry(operand_ref.val).or_insert_with(Vec::new).push(opr);
                }
            }

            // Find dead operations (operations whose results are never used)
            let ops_to_remove: Vec<Opr> = region
                .op_order
                .iter()
                .filter(|&&opr| {
                    if let Some(op) = region.get_op(opr) {
                        // Check if any result is used
                        !op.results
                            .iter()
                            .any(|&result| value_uses.contains_key(&result))
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
