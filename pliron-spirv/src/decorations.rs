use core::hash::Hash;

use alloc::boxed::Box;
use derive_new::new;
use pliron::{
    attribute::{AttrObj, AttributeDict},
    combine::{Parser, attempt, choice, optional, parser::char::char},
    irfmt::parsers,
    parsable::{Parsable, ParseResult, StateStream, parser_combinator},
    printable::Printable,
};

use crate::prelude::*;

pub use crate::autogen_decorations::*;

#[format]
#[derive(new, Clone, Debug)]
pub struct DecorationInfo {
    pub decoration: Decoration,
    pub value: AttrObj,
}

impl DecorationInfo {
    pub fn unit(decoration: Decoration) -> Self {
        Self::new(decoration, Box::new(UnitAttr::new()))
    }
}

impl Hash for DecorationInfo {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.decoration.hash(state);
        self.value.hash(state);
    }
}

impl Eq for DecorationInfo {}
impl PartialEq for DecorationInfo {
    fn eq(&self, other: &Self) -> bool {
        self.decoration == other.decoration && self.value.eq_attr(&*other.value)
    }
}

impl DecorationInfo {
    pub fn decoration_key(&self) -> &'static Identifier {
        self.decoration.decoration_key()
    }
}

pub trait DecorationExt {
    fn decoration_key(&self) -> &'static Identifier;
}

#[inline(never)]
pub fn all_decorations_for_op(
    op: &dyn Op,
    ctx: &Context,
    builder: &mut PlironBuilder,
) -> Result<Vec<(Decoration, Vec<Operand>)>> {
    let op = op.get_operation().deref(ctx);
    all_decorations(&op.attributes, ctx, builder)
}

pub fn all_decorations(
    attrs: &AttributeDict,
    ctx: &Context,
    builder: &mut PlironBuilder,
) -> Result<Vec<(Decoration, Vec<Operand>)>> {
    let mut out = Vec::new();
    for (key, value) in attrs.0.iter() {
        if let Some(decoration) = decoration_for_key(key) {
            let info = DecorationInfo::new(decoration, value.clone());
            out.push((decoration, info.as_operands(ctx, builder)?));
        }
    }
    Ok(out)
}

pub fn print_decorations(ctx: &Context, attrs: &AttributeDict, f: &mut dyn core::fmt::Write) -> core::fmt::Result {
    let mut decorations = vec![];
    for (key, value) in attrs.0.iter() {
        if decoration_for_key(key).is_some() {
            let key = key.disp(ctx).to_string();
            let key = key.strip_prefix("spirv_decoration_").unwrap();
            if value.is::<UnitAttr>() {
                decorations.push(key.to_string())
            } else if let Some(lit) = value.downcast_ref::<LiteralIntegerAttr>() {
                decorations.push(alloc::format!("{key}: {}", lit.0))
            } else if let Some(lit) = value.downcast_ref::<LiteralStringAttr>() {
                decorations.push(alloc::format!("{key}: {}", lit.0))
            } else {
                decorations.push(alloc::format!("{key}: {}", value.disp(ctx)))
            }
        }
    }
    if !decorations.is_empty() {
        write!(f, "{{{}}}", decorations.join(", "))?;
    }
    Ok(())
}

/// Parses a `{key, key: value, ...}` decoration list, prefixing each key with `spirv_decoration_`.
///
/// # Errors
/// Returns a parse error if the input is not a well-formed decoration list.
///
/// # Panics
/// Never in practice: prefixing a parsed identifier always yields a valid identifier.
pub fn decorations_parse<'a>(state_stream: &mut StateStream<'a>, _: ()) -> ParseResult<'a, AttributeDict> {
    let value_parse = choice!(
        attempt(parsers::int_parser().map(|val| -> AttrObj { Box::new(LiteralIntegerAttr::new(val)) })),
        attempt(parsers::quoted_string_parser().map(|val| -> AttrObj { Box::new(LiteralStringAttr::new(val)) })),
        parsers::attr_parser()
    );
    let decoration_parse = (
        Identifier::parser(()),
        optional(attempt(parsers::spaced(char(':'))).with(value_parse)),
    )
        .map(|(key, value)| {
            let key = Identifier::try_new(alloc::format!("spirv_decoration_{key}")).unwrap();
            (key, value.unwrap_or_else(|| Box::new(UnitAttr::new())))
        });
    parsers::delimited_list_parser('{', '}', ',', decoration_parse)
        .map(|entries| AttributeDict(entries.into_iter().collect()))
        .parse_stream(state_stream)
        .into()
}

pub fn decorations_parser<'a>() -> Box<dyn Parser<StateStream<'a>, Output = AttributeDict, PartialState = ()> + 'a> {
    parser_combinator(decorations_parse, ())
}
