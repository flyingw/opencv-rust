use std::borrow::Cow;
use std::collections::HashMap;

use clang::EntityKind;
use once_cell::sync::Lazy;

use super::element::{DefaultRustNativeElement, RustElement};
use super::RustNativeGeneratedElement;
use crate::constant::ValueKind;
use crate::debug::NameDebug;
use crate::type_ref::{FishStyle, NameStyle};
use crate::{settings, CompiledInterpolation, Const, EntityElement, StrExt, SupportedModule};

impl RustElement for Const<'_> {
	fn rust_module(&self) -> SupportedModule {
		DefaultRustNativeElement::rust_module(self.entity())
	}

	fn rust_name(&self, style: NameStyle) -> Cow<str> {
		let mut out = DefaultRustNativeElement::rust_name(self, self.entity(), style);
		if let Some(without_suffix) = out.strip_suffix("_OCVRS_OVERRIDE") {
			let new_len = without_suffix.len();
			out.truncate(new_len);
		}
		out.into()
	}

	fn rendered_doc_comment(&self, comment_marker: &str, opencv_version: &str) -> String {
		DefaultRustNativeElement::rendered_doc_comment(self.entity(), comment_marker, opencv_version)
	}
}

impl RustNativeGeneratedElement for Const<'_> {
	fn element_safe_id(&self) -> String {
		format!("{}-{}", self.rust_module().opencv_name(), self.rust_name(NameStyle::decl()))
	}

	fn gen_rust(&self, opencv_version: &str) -> String {
		static RUST_TPL: Lazy<CompiledInterpolation> = Lazy::new(|| include_str!("tpl/const/rust.tpl.rs").compile_interpolation());

		let parent_is_class = self
			.entity()
			.get_lexical_parent()
			.is_some_and(|p| matches!(p.get_kind(), EntityKind::ClassDecl | EntityKind::StructDecl));
		let name = if parent_is_class {
			self.rust_leafname(FishStyle::No)
		} else {
			self.rust_name(NameStyle::decl())
		};

		if let Some(value) = self.value() {
			let typ = match settings::CONST_TYPE_OVERRIDE.get(name.as_ref()).unwrap_or(&value.kind) {
				ValueKind::Integer => "i32",
				ValueKind::UnsignedInteger => "u32",
				ValueKind::Usize => "usize",
				ValueKind::Float => "f32",
				ValueKind::Double => "f64",
				ValueKind::String => "&str",
			};
			RUST_TPL.interpolate(&HashMap::from([
				("doc_comment", Cow::Owned(self.rendered_doc_comment("///", opencv_version))),
				("debug", self.get_debug().into()),
				("name", name),
				("type", typ.into()),
				("value", value.to_string().into()),
			]))
		} else {
			"".to_string()
		}
	}
}
