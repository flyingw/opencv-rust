use std::borrow::Cow;
use std::collections::HashMap;

use once_cell::sync::Lazy;

use super::element::{DefaultRustNativeElement, RustElement};
use super::RustNativeGeneratedElement;
use crate::debug::NameDebug;
use crate::type_ref::{FishStyle, NameStyle};
use crate::{CompiledInterpolation, EntityElement, Enum, StrExt, SupportedModule};

impl RustElement for Enum<'_> {
	fn rust_module(&self) -> SupportedModule {
		DefaultRustNativeElement::rust_module(self.entity())
	}

	fn rust_name(&self, style: NameStyle) -> Cow<str> {
		DefaultRustNativeElement::rust_name(self, self.entity(), style).into()
	}

	fn rendered_doc_comment(&self, comment_marker: &str, opencv_version: &str) -> String {
		DefaultRustNativeElement::rendered_doc_comment(self.entity(), comment_marker, opencv_version)
	}
}

impl RustNativeGeneratedElement for Enum<'_> {
	fn element_safe_id(&self) -> String {
		format!("{}-{}", self.rust_module().opencv_name(), self.rust_name(NameStyle::decl()))
	}

	fn gen_rust(&self, opencv_version: &str) -> String {
		static ENUM_TPL: Lazy<CompiledInterpolation> = Lazy::new(|| include_str!("tpl/enum/enum.tpl.rs").compile_interpolation());

		static CONST_TPL_SRC: &str = include_str!("tpl/enum/const.tpl.rs");
		static CONST_TPL: Lazy<CompiledInterpolation> = Lazy::new(|| CONST_TPL_SRC.compile_interpolation());

		static CONST_IGNORED_TPL_SRC: &str = include_str!("tpl/enum/const_ignored.tpl.rs");
		static CONST_IGNORED_TPL: Lazy<CompiledInterpolation> = Lazy::new(|| CONST_IGNORED_TPL_SRC.compile_interpolation());

		static FROM_CONST_TPL_SRC: &str = include_str!("tpl/enum/from_const.tpl.rs");
		static FROM_CONST_TPL: Lazy<CompiledInterpolation> = Lazy::new(|| FROM_CONST_TPL_SRC.compile_interpolation());

		static FROM_CONST_IGNORED_TPL_SRC: &str = include_str!("tpl/enum/from_const_ignored.tpl.rs");
		static FROM_CONST_IGNORED_TPL: Lazy<CompiledInterpolation> =
			Lazy::new(|| FROM_CONST_IGNORED_TPL_SRC.compile_interpolation());

		let consts = self.consts();

		let mut enum_consts = String::with_capacity(consts.len() * (CONST_IGNORED_TPL_SRC.len().max(CONST_TPL_SRC.len())) + 32);
		let mut from_consts =
			String::with_capacity(consts.len() * (FROM_CONST_IGNORED_TPL_SRC.len().max(FROM_CONST_TPL_SRC.len())) + 32);

		let mut generated_values = HashMap::<String, Cow<str>>::with_capacity(consts.len());
		for c in &consts {
			let name = c.rust_leafname(FishStyle::No);
			let value = c.value().expect("Can't get value of enum variant").to_string();
			let duplicate_name = generated_values.get(&value).map(|s| s.as_ref());
			let (enum_const_tpl, from_const_tpl) = if duplicate_name.is_some() {
				(&CONST_IGNORED_TPL, &FROM_CONST_IGNORED_TPL)
			} else {
				(&CONST_TPL, &FROM_CONST_TPL)
			};
			let comment_marker = if duplicate_name.is_some() {
				"//"
			} else {
				"///"
			};
			let doc_comment = c.rendered_doc_comment(comment_marker, opencv_version);

			let inter_vars = HashMap::from([
				("name", name.as_ref()),
				("value", value.as_str()),
				("doc_comment", &doc_comment),
				("duplicate_name", duplicate_name.unwrap_or("")),
			]);
			enum_const_tpl.interpolate_into(&mut enum_consts, &inter_vars);
			from_const_tpl.interpolate_into(&mut from_consts, &inter_vars);

			generated_values.insert(value, name);
		}

		ENUM_TPL.interpolate(&HashMap::from([
			("rust_local", self.rust_name(NameStyle::decl()).as_ref()),
			("rust_full", &self.rust_name(NameStyle::ref_())),
			("doc_comment", &self.rendered_doc_comment("///", opencv_version)),
			("debug", &self.get_debug()),
			("enum_consts", &enum_consts),
			("from_consts", &from_consts),
		]))
	}
}
