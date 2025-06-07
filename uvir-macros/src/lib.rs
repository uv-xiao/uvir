use proc_macro::TokenStream;

mod op_derive;
mod type_derive;

#[proc_macro_derive(Op, attributes(operation, _def, _use, _attr, _region))]
pub fn derive_op(input: TokenStream) -> TokenStream {
    op_derive::derive_op(input)
}

#[proc_macro_derive(DialectType, attributes(dialect_type))]
pub fn derive_dialect_type(input: TokenStream) -> TokenStream {
    type_derive::derive_dialect_type(input)
}