use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

pub fn derive_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Parse the operation attribute
    let operation_attr = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("operation"))
        .expect("Missing #[operation] attribute");

    let (dialect, op_name, traits) = parse_operation_attr(operation_attr);

    // Parse struct fields to find defs, uses, attrs, and regions
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("Op must be a struct with named fields"),
        },
        _ => panic!("Op must be a struct"),
    };

    let mut defs = Vec::new();
    let mut uses = Vec::new();
    let mut attrs = Vec::new();
    let mut regions = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_type = &field.ty;

        for attr in &field.attrs {
            if attr.path().is_ident("_def") {
                defs.push((field_name.clone(), field_type.clone(), parse_def_attr(attr)));
            } else if attr.path().is_ident("_use") {
                uses.push((field_name.clone(), field_type.clone()));
            } else if attr.path().is_ident("_attr") {
                attrs.push((field_name.clone(), field_type.clone()));
            } else if attr.path().is_ident("_region") {
                regions.push((field_name.clone(), field_type.clone()));
            }
        }
    }

    // Generate static OpInfo
    let info_name = syn::Ident::new(
        &format!("{}_INFO", name.to_string().to_uppercase()),
        name.span(),
    );
    let verify_fn = syn::Ident::new(
        &format!("{}_verify", name.to_string().to_lowercase()),
        name.span(),
    );
    let parse_fn = syn::Ident::new(
        &format!("{}_parse", name.to_string().to_lowercase()),
        name.span(),
    );
    let print_fn = syn::Ident::new(
        &format!("{}_print", name.to_string().to_lowercase()),
        name.span(),
    );

    let traits_array = traits.iter().map(|t| quote! { #t }).collect::<Vec<_>>();

    let num_operands = uses.len();
    let num_results = defs.len();

    // Generate conversion functions
    let into_op_data_operands = uses.iter().map(|(u, _)| quote! { self.#u });
    let into_op_data_results = defs.iter().map(|(d, _, _)| quote! { self.#d });
    let into_op_data_regions = regions.iter().map(|(r, _)| quote! { self.#r });

    // Generate attribute insertion code
    let attr_insertions = attrs.iter().map(|(name, ty)| {
        let name_str = name.to_string();
        // Check if the type is already Attribute
        if let syn::Type::Path(type_path) = ty {
            if type_path.path.segments.last().map(|s| s.ident.to_string())
                == Some("Attribute".to_string())
            {
                quote! {
                    {
                        let key = ctx.intern_string(#name_str);
                        attributes.push((key, self.#name.clone()));
                    }
                }
            } else {
                // For other types, we need to convert to Attribute
                // For now, we'll just support basic types
                quote! {
                    {
                        let key = ctx.intern_string(#name_str);
                        // TODO: Implement proper conversion based on type
                        attributes.push((key, uvir::attribute::Attribute::Integer(0)));
                    }
                }
            }
        } else {
            quote! {}
        }
    });

    let from_op_data_uses = uses.iter().enumerate().map(|(i, (u, _))| {
        quote! { #u: op.operands[#i].val }
    });
    let from_op_data_defs = defs.iter().enumerate().map(|(i, (d, _, _))| {
        quote! { #d: op.results[#i] }
    });
    let from_op_data_attrs = attrs.iter().map(|(a, ty)| {
        let a_str = a.to_string();
        // Check if the type is already Attribute
        if let syn::Type::Path(type_path) = ty {
            if type_path.path.segments.last().map(|s| s.ident.to_string())
                == Some("Attribute".to_string())
            {
                quote! {
                    #a: op.attributes.iter()
                        .find(|(k, _)| ctx.get_string(*k) == Some(#a_str))
                        .map(|(_, v)| v.clone())
                        .unwrap_or(uvir::attribute::Attribute::Integer(0))
                }
            } else {
                // For other types, we need to convert from Attribute
                quote! {
                    #a: {
                        let attr = op.attributes.iter()
                            .find(|(k, _)| ctx.get_string(*k) == Some(#a_str))
                            .map(|(_, v)| v);
                        // TODO: Implement conversion from Attribute to field type
                        Default::default()
                    }
                }
            }
        } else {
            quote! { #a: Default::default() }
        }
    });
    let from_op_data_regions = regions.iter().enumerate().map(|(i, (r, _))| {
        quote! { #r: op.regions[#i] }
    });

    let expanded = quote! {
        // Generate static OpInfo
        static #info_name: uvir::ops::OpInfo = uvir::ops::OpInfo {
            dialect: #dialect,
            name: #op_name,
            traits: &[#(#traits_array),*],
            verify: #verify_fn,
            parse: #parse_fn,
            print: #print_fn,
        };

        // Register with inventory
        uvir::inventory::submit!(&#info_name);

        // Verification function
        fn #verify_fn(op: &uvir::ops::OpData) -> uvir::error::Result<()> {
            // Basic verification: check operand and result counts
            if op.operands.len() != #num_operands {
                return Err(uvir::error::Error::VerificationError(
                    format!("Expected {} operands, found {}", #num_operands, op.operands.len())
                ));
            }
            if op.results.len() != #num_results {
                return Err(uvir::error::Error::VerificationError(
                    format!("Expected {} results, found {}", #num_results, op.results.len())
                ));
            }

            // Note: Type constraints and traits that require type information
            // are verified separately in the verification module with Context access

            Ok(())
        }

        // Parse function
        fn #parse_fn(parser: &mut uvir::parser::Parser) -> uvir::error::Result<uvir::ops::OpData> {
            // TODO: Implement parsing
            Err(uvir::error::Error::ParseError("Operation parsing not yet implemented".to_string()))
        }

        // Print function
        fn #print_fn(op: &uvir::ops::OpData, printer: &mut uvir::printer::Printer) -> uvir::error::Result<()> {
            // Basic printing
            printer.print(&format!("{}.{}", #dialect, #op_name))?;

            // Print results
            if !op.results.is_empty() {
                printer.print(" ")?;
                for (i, result) in op.results.iter().enumerate() {
                    if i > 0 {
                        printer.print(", ")?;
                    }
                    printer.print(&format!("%{:?}", result))?;
                }
                printer.print(" =")?;
            }

            // Print operands
            if !op.operands.is_empty() {
                printer.print(" ")?;
                for (i, operand) in op.operands.iter().enumerate() {
                    if i > 0 {
                        printer.print(", ")?;
                    }
                    printer.print(&format!("%{:?}", operand))?;
                }
            }

            // Print attributes
            if !op.attributes.is_empty() {
                printer.print(" {")?;
                for (i, (key, attr)) in op.attributes.iter().enumerate() {
                    if i > 0 {
                        printer.print(", ")?;
                    }
                    // TODO: Print attribute properly
                    printer.print(&format!("{} = ...", key))?;
                }
                printer.print("}")?;
            }

            Ok(())
        }

        impl #name {
            pub fn into_op_data(mut self, ctx: &mut uvir::context::Context, region: uvir::region::RegionId) -> uvir::ops::OpData {
                // Create attributes vector with proper conversions
                let mut attributes = uvir::smallvec::SmallVec::new();
                #(#attr_insertions)*

                // Extract fields after attributes (which may need cloning)
                let operands = uvir::smallvec::smallvec![#(ctx.make_value_ref(region, #into_op_data_operands)),*];
                let results = uvir::smallvec::smallvec![#(#into_op_data_results),*];
                let regions = uvir::smallvec::smallvec![#(#into_op_data_regions),*];

                uvir::ops::OpData {
                    info: &#info_name,
                    operands,
                    results,
                    attributes,
                    regions,
                    custom_data: uvir::ops::OpStorage::from_value(self),
                }
            }

            pub fn from_op_data(op: &uvir::ops::OpData, ctx: &uvir::Context) -> Self {
                // Extract fields from OpData
                let mut uses_iter = op.operands.iter();
                let mut defs_iter = op.results.iter();

                Self {
                    #(#from_op_data_uses,)*
                    #(#from_op_data_defs,)*
                    #(#from_op_data_attrs,)*
                    #(#from_op_data_regions,)*
                }
            }

            pub fn op_info() -> &'static uvir::ops::OpInfo {
                &#info_name
            }

            pub fn info() -> &'static uvir::ops::OpInfo {
                &#info_name
            }
        }

        impl uvir::ops::Op for #name {
            fn info(&self) -> &'static uvir::ops::OpInfo {
                &#info_name
            }
        }
    };

    TokenStream::from(expanded)
}

fn parse_operation_attr(attr: &syn::Attribute) -> (&'static str, &'static str, Vec<&'static str>) {
    let mut dialect = None;
    let mut name = None;
    let mut traits = Vec::new();

    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("dialect") {
            let value = meta.value()?;
            let s: syn::LitStr = value.parse()?;
            dialect = Some(Box::leak(s.value().into_boxed_str()) as &'static str);
        } else if meta.path.is_ident("name") {
            let value = meta.value()?;
            let s: syn::LitStr = value.parse()?;
            name = Some(Box::leak(s.value().into_boxed_str()) as &'static str);
        } else if meta.path.is_ident("traits") {
            let value = meta.value()?;
            let s: syn::LitStr = value.parse()?;
            let traits_str = s.value();
            for trait_name in traits_str.split(',') {
                let trait_name = trait_name.trim();
                if !trait_name.is_empty() {
                    traits.push(Box::leak(trait_name.to_string().into_boxed_str()) as &'static str);
                }
            }
        }
        Ok(())
    })
    .expect("Failed to parse operation attribute");

    (
        dialect.expect("Missing dialect in operation attribute"),
        name.expect("Missing name in operation attribute"),
        traits,
    )
}

fn parse_def_attr(attr: &syn::Attribute) -> Option<String> {
    // Parse type constraint from _def attribute like #[_def(ty = "T")]
    let mut type_constraint = None;

    if attr.path().is_ident("_def") {
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("ty") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                type_constraint = Some(s.value());
            }
            Ok(())
        });
    }

    type_constraint
}
