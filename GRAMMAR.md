# MLIR Grammar Reference

This document provides a comprehensive summary of the MLIR grammar based on the [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/). It includes the complete grammar specification while noting uvir-specific limitations (no blocks/branches, only IsolatedFromAbove regions).

## EBNF Notation

```
alternation ::= expr0 | expr1 | expr2  // Either expr0 or expr1 or expr2.
sequence    ::= expr0 expr1 expr2      // Sequence of expr0 expr1 expr2.
repetition0 ::= expr*  // 0 or more occurrences.
repetition1 ::= expr+  // 1 or more occurrences.
optionality ::= expr?  // 0 or 1 occurrence.
grouping    ::= (expr) // Everything inside parens is grouped together.
literal     ::= `abcd` // Matches the literal `abcd`.
```

## Common Syntax

### Lexical Elements

```
// Character classes
digit     ::= [0-9]
hex_digit ::= [0-9a-fA-F]
letter    ::= [a-zA-Z]
id-punct  ::= [$._-]

// Literals
integer-literal ::= decimal-literal | hexadecimal-literal
decimal-literal ::= digit+
hexadecimal-literal ::= `0x` hex_digit+
float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
              | `0x` hex_digit+ `p` [+-]? digit+
              | (inf | nan | inf.0 | nan.0)
string-literal  ::= `"` (char | escape)* `"`
char           ::= [^"\n\f\v\r\\]
escape         ::= `\\` [nfrtvb"'\\]
                 | `\\x` hex_digit hex_digit
                 | `\\` digit digit digit
```

### Comments

MLIR supports BCPL-style comments starting with `//` and continuing until end of line.

### Top Level Productions

```
// Top level production
top-level-module ::= module-op
module-op ::= `module` module-attributes? region

// Simplified for parsing without explicit module
toplevel ::= (operation | attribute-alias-def | type-alias-def)*
```

### Identifiers and Keywords

```
// Identifiers
bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
alias-name ::= bare-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
value-id-list ::= value-id (`,` value-id)*

// Value uses
value-use ::= value-id
value-use-list ::= value-use (`,` value-use)*
```

## Operations

```
operation             ::= op-result-list? (generic-operation | custom-operation)
                          trailing-location?
generic-operation     ::= string-literal `(` value-use-list? `)` successor-list?
                          dictionary-properties? region-list? dictionary-attribute?
                          `:` function-type
custom-operation      ::= bare-id custom-operation-format
op-result-list        ::= op-result (`,` op-result)* `=`
op-result             ::= value-id (`:` integer-literal)?
successor-list        ::= `[` successor (`,` successor)* `]`
successor             ::= caret-id (`:` block-arg-list)?
dictionary-properties ::= `<` dictionary-attribute `>`
region-list           ::= `(` region (`,` region)* `)`
dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
trailing-location     ::= `loc` `(` location `)`

// Custom operation format is dialect-specific and flexible
```

**Note:** uvir excludes successor-list and block operations, so successor-list will always be empty.

### Location Information

```
location ::= filelinecol-location | name-location | callsite-location | fused-location | unknown-location
filelinecol-location ::= string-literal `:` integer-literal `:` integer-literal
name-location ::= string-literal (`(` location `)`)?  
callsite-location ::= location `at` location
fused-location ::= `fused` (`<` attribute-value `>`)? `[` location (`,` location)* `]`
unknown-location ::= `unknown`
```

## Regions (Simplified for uvir)

uvir only supports IsolatedFromAbove regions with structured control flow (no blocks/branches).

```
region      ::= `{` block* `}`
block       ::= block-label? operation*
block-label ::= caret-id block-arg-list? `:`
caret-id    ::= `^` suffix-id  
block-arg-list ::= `(` block-arg (`,` block-arg)* `)`
block-arg   ::= value-id `:` type
```

**Note:** Since uvir doesn't support multiple blocks, regions are simplified to contain only operations directly. The block syntax above is included for completeness but uvir regions will only have a single implicit entry block with no label.

## Type System

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// Type variants
function-type ::= (type | type-list-parens) `->` (type | type-list-parens)

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type
ssa-use ::= value-use

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*
```

### Type Aliases

```
type-alias-def ::= `!` alias-name `=` type
type-alias ::= `!` alias-name
```

### Dialect Types

```
dialect-namespace ::= bare-id

dialect-type ::= `!` (opaque-dialect-type | pretty-dialect-type)
opaque-dialect-type ::= dialect-namespace dialect-type-body
pretty-dialect-type ::= dialect-namespace `.` pretty-dialect-type-lead-ident
                                              dialect-type-body?
pretty-dialect-type-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

dialect-type-body ::= `<` dialect-type-contents+ `>`
dialect-type-contents ::= dialect-type-body
                            | `(` dialect-type-contents+ `)`
                            | `[` dialect-type-contents+ `]`
                            | `{` dialect-type-contents+ `}`
                            | [^\[<({\]>)}\0]+
```

### Builtin Types

```
builtin-type ::= integer-type | float-type | index-type | none-type | complex-type
               | memref-type | tensor-type | vector-type

// Signless integer types               
integer-type ::= `i` [1-9][0-9]*

// Signed integer types (less common)
signed-integer-type ::= `si` [1-9][0-9]*

// Unsigned integer types (less common)
unsigned-integer-type ::= `ui` [1-9][0-9]*

// Float types
float-type ::= `f16` | `bf16` | `f32` | `f64` | `f80` | `f128`

// Index type (platform-specific integer)
index-type ::= `index`

// None type (no value)
none-type ::= `none`

// Complex types
complex-type ::= `complex` `<` (integer-type | float-type) `>`

// Container types (not fully supported in uvir)
memref-type ::= `memref` `<` dimension-list type (`,` memory-space)? `>`
tensor-type ::= `tensor` `<` dimension-list type `>`  
vector-type ::= `vector` `<` dimension-list type `>`

dimension-list ::= dimension-list-ranked | dimension-list-unranked
dimension-list-ranked ::= (dimension `x`)*
dimension ::= `?` | integer-literal
dimension-list-unranked ::= `*` `x`
```

## Attributes

```
attribute-entry ::= (bare-id | string-literal) `=` attribute-value
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
```

### Attribute Aliases

```
attribute-alias-def ::= `#` alias-name `=` attribute-value
attribute-alias ::= `#` alias-name
```

### Dialect Attributes

```
dialect-namespace ::= bare-id

dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
opaque-dialect-attribute ::= dialect-namespace dialect-attribute-body
pretty-dialect-attribute ::= dialect-namespace `.` pretty-dialect-attribute-lead-ident
                                              dialect-attribute-body?
pretty-dialect-attribute-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

dialect-attribute-body ::= `<` dialect-attribute-contents+ `>`
dialect-attribute-contents ::= dialect-attribute-body
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^\[<({\]>)}\0]+
```

### Builtin Attributes

```
builtin-attribute ::= affine-map-attribute | affine-set-attribute | array-attribute
                    | bool-attribute | dense-attribute | dictionary-attribute  
                    | float-attribute | integer-attribute | integer-set-attribute
                    | opaque-attribute | sparse-elements-attribute | string-attribute
                    | symbol-ref-attribute | type-attribute | unit-attribute

// Simple attributes
bool-attribute ::= bool-literal
bool-literal ::= `true` | `false`
integer-attribute ::= integer-literal ( `:` (index-type | integer-type) )?
float-attribute ::= float-literal ( `:` float-type )?
string-attribute ::= string-literal ( `:` type )?
type-attribute ::= type
unit-attribute ::= `unit`

// Container attributes
array-attribute ::= `[` (attribute-value (`,` attribute-value)*)? `]`
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`

// Symbol references
symbol-ref-attribute ::= symbol-ref-id

// Dense elements (simplified)
dense-attribute ::= `dense` `<` (constant-literal | array-attribute) `>` `:` shaped-type

// Affine maps and sets (not fully supported in uvir)
affine-map-attribute ::= `affine_map` `<` affine-map `>`
affine-set-attribute ::= `affine_set` `<` affine-set `>`
```

## Examples

### Basic Operations

```mlir
// Generic form - operation name is a string literal
%result = "dialect.operation"(%operand1, %operand2) {attr = "value"} : (i32, i32) -> i32

// Custom syntax - dialect can define custom parsing/printing
%result = dialect.operation %operand1, %operand2 {attr = "value"} : (i32, i32) -> i32

// Operation with regions
"dialect.op_with_region"() ({
  ^bb0:
    %0 = "dialect.inner"() : () -> i32
    "dialect.terminator"(%0) : (i32) -> ()
}) : () -> ()

// Operation with multiple results  
%res:2 = "dialect.multi_result"() : () -> (i32, f32)

// Operation with successor blocks (not supported in uvir)
"cf.br"()[^bb1] : () -> ()
```

### Function Definitions

```mlir
// Function with func dialect
func.func @simple_function(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}

// Function with attributes
func.func @attributed_function(%arg0: f32) -> f32 attributes {visibility = "public"} {
  %0 = arith.mulf %arg0, %arg0 : f32
  func.return %0 : f32
}

// Function with multiple results
func.func @multi_result(%arg0: i32) -> (i32, i32) {
  %0 = arith.constant 42 : i32
  func.return %arg0, %0 : i32, i32
}
```

### Type and Attribute Aliases

```mlir
// Type aliases - defined with ! prefix
!my_int = i32
!matrix = tensor<4x4xf32>
!vec3 = vector<3xf32>

// Using type aliases
func.func @use_type_aliases(%x: !my_int, %y: !matrix) -> !vec3 {
  // ... operations
}

// Attribute aliases - defined with # prefix  
#map = affine_map<(d0, d1) -> (d0 + d1)>
#loc = loc("file.mlir":10:5)
#config = {device = "gpu", threads = 256}

// Using attribute aliases
%result = "dialect.op"() {config = #config} : () -> i32 loc(#loc)
```

### Module Structure

```mlir
// Explicit module with attributes
module @my_module attributes {version = "1.0"} {
  // Module-level attribute and type aliases
  !tensor_type = tensor<?x?xf32>
  #map = affine_map<(d0, d1) -> (d0, d1)>
  
  // Operations within module
  func.func @foo(%arg0: !tensor_type) -> !tensor_type {
    func.return %arg0 : !tensor_type
  }
}

// Implicit module (when module op is omitted)
func.func @bar() {
  func.return
}
```

## uvir-Specific Limitations

1. **No Block Operations**: uvir does not support block labels (`^bb0:`) or explicit block arguments
2. **No Branch Operations**: No `cf.br`, `cf.cond_br`, or similar control flow operations
3. **Structured Control Flow Only**: Only structured operations like `scf.for`, `scf.if`, etc.
4. **IsolatedFromAbove Regions**: All regions are isolated from their containing scope
5. **Single Block Regions**: Each region contains at most one implicit block

## Additional Grammar Elements

### Dialect Operations

Each dialect can define custom operation syntax. Common patterns:

```mlir
// Arithmetic dialect
%sum = arith.addi %a, %b : i32
%prod = arith.mulf %x, %y : f32

// SCF (Structured Control Flow) dialect  
scf.for %i = %c0 to %c10 step %c1 {
  // loop body
}

scf.if %condition {
  // then region
} else {
  // else region
}

// Affine dialect
affine.for %i = 0 to 10 {
  // affine loop body
}
```

### Special Syntax Elements

```
// Variadic operands/results
variadic-list ::= value-use (`,` value-use)*

// Optional groups in custom syntax
optional-group ::= `(` elements `)` `?`

// Custom directives for operation printing
custom-directive ::= `custom` `<` directive-name `>` `(` parameters `)`
```

## Compliance Notes

uvir aims to be compatible with the textual MLIR format except for the limitations listed above. This means:

- All type syntax is identical to MLIR
- All attribute syntax is identical to MLIR  
- Operation syntax follows MLIR conventions
- Parsing and printing should round-trip correctly
- Comments are supported using `//` syntax
- String escaping follows MLIR rules
- Location tracking is optional but supported
- Module structure can be implicit or explicit 