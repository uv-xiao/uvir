# MLIR Grammar Reference

This document summarizes the MLIR grammar based on the [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/), excluding Block and Branch operations as specified in the uvir project limitations.

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
// TODO: Clarify the split between lexing (tokens) and parsing (grammar).
digit     ::= [0-9]
hex_digit ::= [0-9a-fA-F]
letter    ::= [a-zA-Z]
id-punct  ::= [$._-]

integer-literal ::= decimal-literal | hexadecimal-literal
decimal-literal ::= digit+
hexadecimal-literal ::= `0x` hex_digit+
float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
string-literal  ::= `"` [^"\n\f\v\r]* `"`   // TODO: define escaping rules
```

### Comments

MLIR supports BCPL-style comments starting with `//` and continuing until end of line.

### Top Level Productions

```
// Top level production
toplevel := (operation | attribute-alias-def | type-alias-def)*
```

### Identifiers and Keywords

```
// Identifiers
bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
alias-name :: = bare-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
value-id-list ::= value-id (`,` value-id)*
```

## Operations

```
operation             ::= op-result-list? (generic-operation | custom-operation)
                          trailing-location?
generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
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
```

**Note:** uvir excludes successor-list and block operations, so successor-list will always be empty.

## Regions (Simplified for uvir)

uvir only supports IsolatedFromAbove regions with structured control flow (no blocks/branches).

```
region      ::= `{` entry-block? operation* `}`
entry-block ::= operation+
```

**Note:** Since uvir doesn't support multiple blocks, regions are simplified to contain only operations directly.

## Type System

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type
ssa-use ::= value-use

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
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

Common builtin types include:
- Integer types: `i1`, `i8`, `i16`, `i32`, `i64`, etc.
- Unsigned integer types: `ui8`, `ui16`, `ui32`, `ui64`, etc. (though typically written as `i8`, `i16`, etc.)
- Float types: `f16`, `f32`, `f64`, `bf16`
- Index type: `index`
- Function types: `(type-list) -> (type-list)`

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

Common builtin attributes include:
- Integer literals: `42`, `-123`
- Float literals: `3.14`, `1.0e-5`
- String literals: `"hello world"`
- Array attributes: `[1, 2, 3]`
- Dictionary attributes: `{key = value, ...}`
- Type attributes: references to types
- Unit attributes: `unit`

## Examples

### Basic Operation

```mlir
// Generic form
%result = "dialect.operation"(%operand1, %operand2) {attr = "value"} : (i32, i32) -> i32

// Custom syntax (dialect-specific)
%result = dialect.operation %operand1, %operand2 {attr = "value"} : (i32, i32) -> i32
```

### Function Definition (using func dialect)

```mlir
func.func @simple_function(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
```

### Type Aliases

```mlir
!my_int = i32
!matrix = tensor<4x4xf32>

func.func @use_aliases(%x: !my_int, %y: !matrix) -> !my_int {
  // ... operations
}
```

### Attribute Aliases

```mlir
#map = affine_map<(d0, d1) -> (d0 + d1)>

func.func @use_map(%arg0: index, %arg1: index) -> index {
  %result = affine.apply #map(%arg0, %arg1)
  func.return %result : index
}
```

## uvir-Specific Limitations

1. **No Block Operations**: uvir does not support block labels (`^bb0:`) or explicit block arguments
2. **No Branch Operations**: No `cf.br`, `cf.cond_br`, or similar control flow operations
3. **Structured Control Flow Only**: Only structured operations like `scf.for`, `scf.if`, etc.
4. **IsolatedFromAbove Regions**: All regions are isolated from their containing scope
5. **Single Block Regions**: Each region contains at most one implicit block

## Compliance Notes

uvir aims to be compatible with the textual MLIR format except for the limitations listed above. This means:

- All type syntax is identical to MLIR
- All attribute syntax is identical to MLIR  
- Operation syntax follows MLIR conventions
- Parsing and printing should round-trip correctly
- Comments are supported using `//` syntax
- String escaping follows MLIR rules 