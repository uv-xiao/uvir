# egg Integration

uvir provides seamless integration with [egg](https://egraphs-good.github.io/), a library for e-graphs and equality saturation. This enables advanced optimization techniques like equality saturation and extraction.

## Overview

The integration allows bidirectional conversion between uvir IR and egg e-graphs:
- Convert uvir operations to egg expressions
- Apply egg rewrites and extract optimized expressions
- Convert back to uvir IR

## Basic Usage

```rust
use uvir::prelude::*;
use uvir::egg_interop::*;

// Convert a uvir region to an e-graph
let mut egraph = ctx.to_egraph(region_id)?;

// Apply rewrite rules
let rules = arith_rules();
let mut runner = Runner::default()
    .with_egraph(egraph)
    .run(&rules);

// Extract the best program
let best = Extractor::new(&runner.egraph, AstSize).find_best(root);

// Convert back to uvir
let optimized_region = Context::from_egraph(&runner.egraph, best)?;
```

## Defining egg Languages

Define an egg language that corresponds to your dialect:

```rust
use egg::{define_language, Id};

define_language! {
    pub enum ArithLang {
        // Constants
        "const" = Const([Id; 0], i64),
        
        // Binary operations
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Sub([Id; 2]),
        "/" = Div([Id; 2]),
        
        // Comparisons
        "==" = Eq([Id; 2]),
        "<" = Lt([Id; 2]),
        
        // Variables
        Var(String),
    }
}
```

## Operation Conversion

### Implementing IntoEgg

```rust
impl IntoEgg for arith::AddIOp {
    type EggLang = ArithLang;
    
    fn into_egg(&self, conv: &mut UvirToEgg) -> Id {
        let lhs = conv.convert_value(self.lhs);
        let rhs = conv.convert_value(self.rhs);
        let node = ArithLang::Add([lhs, rhs]);
        conv.add_node(node)
    }
}

impl IntoEgg for arith::ConstantOp {
    type EggLang = ArithLang;
    
    fn into_egg(&self, conv: &mut UvirToEgg) -> Id {
        match &self.value {
            Attribute::Integer(i) => {
                let node = ArithLang::Const([], *i);
                conv.add_node(node)
            }
            _ => panic!("Unsupported constant type"),
        }
    }
}
```

### Implementing FromEgg

```rust
impl FromEgg for arith::AddIOp {
    type EggLang = ArithLang;
    
    fn from_egg(enode: &ArithLang, conv: &mut EggToUvir) -> Result<OpData> {
        match enode {
            ArithLang::Add([lhs, rhs]) => {
                let lhs_val = conv.convert_id(*lhs)?;
                let rhs_val = conv.convert_id(*rhs)?;
                let result_type = conv.ctx.get_value_type(lhs_val);
                let result = conv.ctx.create_value(result_type);
                
                Ok(arith::AddIOp {
                    result,
                    lhs: lhs_val,
                    rhs: rhs_val,
                }.into_op_data(conv.ctx))
            }
            _ => Err(Error::UnexpectedENode),
        }
    }
}
```

## Conversion Context

The conversion maintains mappings between uvir values and egg IDs:

```rust
pub struct UvirToEgg<'a> {
    ctx: &'a Context,
    egraph: &'a mut EGraph<ArithLang, ()>,
    val_to_id: HashMap<Val, Id>,
}

impl<'a> UvirToEgg<'a> {
    pub fn convert_value(&mut self, val: Val) -> Id {
        if let Some(&id) = self.val_to_id.get(&val) {
            return id;
        }
        
        // Find defining operation
        if let Some(op) = self.ctx.get_defining_op(val) {
            let id = self.convert_op(op);
            self.val_to_id.insert(val, id);
            id
        } else {
            // Block argument or external value
            let var = ArithLang::Var(format!("v{}", val.0));
            self.add_node(var)
        }
    }
    
    pub fn add_node(&mut self, node: ArithLang) -> Id {
        self.egraph.add(node)
    }
}
```

## Rewrite Rules

Define rewrite rules for egg:

```rust
use egg::{rewrite as rw, Rewrite};

pub fn arith_rules() -> Vec<Rewrite<ArithLang, ()>> {
    vec![
        // Associativity
        rw!("add-assoc"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("mul-assoc"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        
        // Commutativity
        rw!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rw!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
        
        // Identity
        rw!("add-zero"; "(+ ?a (const 0))" => "?a"),
        rw!("mul-one"; "(* ?a (const 1))" => "?a"),
        
        // Zero annihilation
        rw!("mul-zero"; "(* ?a (const 0))" => "(const 0)"),
        
        // Distribution
        rw!("distrib"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        
        // Constant folding
        rw!("fold-add"; "(+ (const ?a) (const ?b))" => "(const (+ ?a ?b))"),
        rw!("fold-mul"; "(* (const ?a) (const ?b))" => "(const (* ?a ?b))"),
    ]
}
```

## Cost Functions

Define custom cost functions for extraction:

```rust
use egg::{CostFunction, Language};

struct ArithCost;

impl CostFunction<ArithLang> for ArithCost {
    type Cost = usize;
    
    fn cost<C>(&mut self, enode: &ArithLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let base_cost = match enode {
            ArithLang::Const(..) => 1,
            ArithLang::Var(..) => 1,
            ArithLang::Add(..) => 2,
            ArithLang::Sub(..) => 2,
            ArithLang::Mul(..) => 3,
            ArithLang::Div(..) => 10,
            _ => 1,
        };
        
        enode.fold(base_cost, |sum, id| sum + costs(id))
    }
}
```

## Complete Example

Here's a complete example optimizing arithmetic expressions:

```rust
use uvir::prelude::*;
use uvir::egg_interop::*;
use egg::{EGraph, Runner, Extractor, AstSize};

fn optimize_arithmetic(ctx: &mut Context, func: FuncOp) -> Result<()> {
    // Convert function body to e-graph
    let body_region = func.body;
    let mut egraph = EGraph::default();
    let mut converter = UvirToEgg::new(ctx, &mut egraph);
    
    // Convert all operations
    let mut roots = Vec::new();
    for op in ctx.region_ops(body_region) {
        let id = converter.convert_op(op);
        roots.push(id);
    }
    
    // Run equality saturation
    let rules = arith_rules();
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(20)
        .with_time_limit(Duration::from_secs(5))
        .run(&rules);
    
    println!("Performed {} iterations", runner.iterations.len());
    println!("E-graph size: {} enodes, {} eclasses", 
             runner.egraph.total_size(),
             runner.egraph.number_of_classes());
    
    // Extract optimized expressions
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let mut back_converter = EggToUvir::new(ctx);
    
    // Clear original region
    ctx.clear_region(body_region)?;
    
    // Convert back optimized operations
    for root in roots {
        let (best_cost, best_expr) = extractor.find_best(root);
        println!("Optimized from cost {} to {}", 
                 AstSize.cost(&runner.egraph[root].nodes[0], |_| 0),
                 best_cost);
        
        back_converter.convert_expr(&runner.egraph, best_expr)?;
    }
    
    Ok(())
}
```

## Advanced Techniques

### Conditional Rewrites

```rust
use egg::{Condition, Pattern, Var};

struct IsPositiveConst;

impl Condition<ArithLang, ()> for IsPositiveConst {
    fn check(&self, egraph: &mut EGraph<ArithLang, ()>, eclass: Id, subst: &Subst) -> bool {
        if let Some(c) = &egraph[eclass].nodes.iter().find_map(|n| {
            if let ArithLang::Const([], i) = n {
                Some(*i)
            } else {
                None
            }
        }) {
            *c > 0
        } else {
            false
        }
    }
}

// Use in rewrite
rw!("div-by-self"; "(/ ?a ?a)" => "(const 1)" if IsPositiveConst),
```

### E-class Analysis

```rust
use egg::{Analysis, EGraph};

#[derive(Default)]
struct ConstantAnalysis;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstantData {
    constant: Option<i64>,
}

impl Analysis<ArithLang> for ConstantAnalysis {
    type Data = ConstantData;
    
    fn make(egraph: &EGraph<ArithLang, Self::Data>, enode: &ArithLang) -> Self::Data {
        let constant = match enode {
            ArithLang::Const([], i) => Some(*i),
            ArithLang::Add([a, b]) => {
                let a_data = &egraph[*a].data;
                let b_data = &egraph[*b].data;
                match (a_data.constant, b_data.constant) {
                    (Some(a), Some(b)) => Some(a + b),
                    _ => None,
                }
            }
            _ => None,
        };
        ConstantData { constant }
    }
    
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if to.constant != from.constant {
            to.constant = None;
            DidMerge(true, false)
        } else {
            DidMerge(false, false)
        }
    }
}
```

### Integration with Passes

```rust
pub struct EqualitySaturationPass {
    rules: Vec<Rewrite<ArithLang, ()>>,
    iter_limit: usize,
}

impl Pass for EqualitySaturationPass {
    fn name(&self) -> &str { "equality-saturation" }
    
    fn run(&mut self, ctx: &mut Context) -> Result<PassResult> {
        let mut changed = false;
        let mut stats = HashMap::new();
        
        // Process each function
        for func in ctx.functions() {
            let initial_cost = count_operations(ctx, func.body);
            
            optimize_arithmetic(ctx, func)?;
            
            let final_cost = count_operations(ctx, func.body);
            if final_cost < initial_cost {
                changed = true;
                stats.insert(
                    format!("{}_reduction", func.name),
                    (initial_cost - final_cost) as u64,
                );
            }
        }
        
        Ok(PassResult { changed, statistics: stats })
    }
}
```

## Performance Tips

1. **Limit e-graph size**: Use iteration and node limits
2. **Choose good cost functions**: Balance accuracy and speed
3. **Prune unnecessary patterns**: Only include relevant rewrites
4. **Use e-class analysis**: Share computed data across equivalences
5. **Batch conversions**: Convert multiple operations together

## Debugging

Enable egg's logging for debugging:

```rust
use log::info;

// In your conversion
info!("Converting operation: {:?}", op);
info!("E-graph state: {} nodes, {} classes", 
      egraph.total_size(), 
      egraph.number_of_classes());

// Visualize e-graph (requires graphviz)
egraph.dot().to_pdf("egraph.pdf").unwrap();
```

## Limitations

- Not all uvir operations map cleanly to egg expressions
- Region-based operations need special handling  
- Type information may be lost during conversion
- Large e-graphs can consume significant memory

## Future Work

- Automatic rule generation from operation semantics
- Typed e-graphs for preserving type information
- Incremental equality saturation
- Integration with other e-graph libraries