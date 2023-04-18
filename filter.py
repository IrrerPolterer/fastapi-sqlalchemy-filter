from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Callable, Iterable, List, Self, Tuple, Type
from uuid import UUID

import humps
from fastapi import Query
from fastapi.params import Query as QueryType
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from sqlalchemy import Column, Select, inspect
from sqlalchemy.orm import DeclarativeBase, Mapper


@dataclasses.dataclass(frozen=True)
class Operator:
    """
    Filter Operator, describing a filter operation.

    Attributes:
    - `description` is used for the auto-generated openapi documentation.
    - `suffix` is the shorthand for this filter operation, used to name the resulting query parameters.
    - `supported_type` is the type (or Union of types) which support this filter operation. This identifies which columns of the data model support this filter operation.
    - `unsupported_type` is the type (or Union of types) which do not support this filter operation. This helps narrow down the exact data types that support this filter operation in situations with inherited types.
    - `input_type` can be used to force a parameter input type for this filter operation. If unset, the query parameter assumes the data type of the associated data model column.
    """

    description: str
    suffix: str
    supported_type: Type
    compare_func: Callable[[Column, Any], Any]

    # optional attributes
    unsupported_type: Type = None
    input_type: Type = None

    def supports_column(self, column: Column) -> bool:
        """
        Returns `True` if given sqlalchemy column object can be filtered with this operator.
        This classification is based entirely on the `supported_type` and `unsupported_type` of the operator.
        """
        column_type = column.type.python_type

        return issubclass(column_type, self.supported_type) and not (
            self.unsupported_type
            and issubclass(column_type, self.unsupported_type)
        )


OPERATORS = (
    Operator(
        description="equal",
        suffix="eq",
        supported_type=int | float | bool | str | UUID,
        compare_func=lambda property, value: property == value,
    ),
    Operator(
        description="non-equal",
        suffix="neq",
        supported_type=int | float | bool | str | UUID,
        compare_func=lambda property, value: property != value,
    ),
    Operator(
        description="lower-than",
        suffix="lt",
        supported_type=int | float | str,
        unsupported_type=Enum,
        compare_func=lambda property, value: property < value,
    ),
    Operator(
        description="lower-than or equal",
        suffix="lte",
        supported_type=int | float | str,
        unsupported_type=Enum,
        compare_func=lambda property, value: property <= value,
    ),
    Operator(
        description="greater-than",
        suffix="gt",
        supported_type=int | float | str,
        unsupported_type=Enum,
        compare_func=lambda property, value: property > value,
    ),
    Operator(
        description="greater-than or equal",
        suffix="gte",
        supported_type=int | float | str,
        unsupported_type=Enum,
        compare_func=lambda property, value: property >= value,
    ),
    Operator(
        description="in-array",
        suffix="in",
        supported_type=int | float | bool | str | UUID,
        input_type=list,
        compare_func=lambda property, value: property.in_(value),
    ),
    Operator(
        description="contains",
        suffix="contains",
        supported_type=str,
        unsupported_type=Enum,
        compare_func=lambda property, value: property.like(f"%{value}%"),
    ),
    Operator(
        description="case-insensitive contains",
        suffix="icontains",
        supported_type=str,
        unsupported_type=Enum,
        compare_func=lambda property, value: property.ilike(f"%{value}%"),
    ),
    Operator(
        description="is-null or not-null",
        suffix="null",
        supported_type=int | float | bool | str | UUID,
        input_type=bool,
        compare_func=lambda property, value: property.is_(None)
        if value
        else property.is_not(None),
    ),
)


@dataclasses.dataclass(frozen=True)
class FilterOption:
    """
    Filter option, representing a possible filtering parameter for a given data model.
    They associate columns of an sqlalchemy data model with applicable filter operations.

    Appropriate fastapi query parameters can be declared using the `query_parameter()` method.
    """

    column: Column
    operator: Operator
    relationship_chain: Iterable[str] = ()

    @property
    def _name_relation_prefix(self) -> str:
        """
        Relationship chain string-formatted as for use in valid python object names
        """
        return (
            f"{'_'.join(self.relationship_chain)}_"
            if self.relationship_chain
            else ""
        )

    @property
    def _alias_relation_prefix(self) -> str:
        """
        Relationship chain string-formatted as for use in camel-cased and dot-separated query option aliases.
        """
        return (
            f"{'.'.join(humps.camelize(x) for x in self.relationship_chain)}."
            if self.relationship_chain
            else ""
        )

    @property
    def column_name(self) -> str:
        """
        Column names formatted as valid python object name.
        """
        return f"{self._name_relation_prefix}{self.column.name}"

    @property
    def column_alias(self) -> str:
        """
        Column names formatted as camel-cased query option aliases.
        """
        return (
            f"{self._alias_relation_prefix}{humps.camelize(self.column.name)}"
        )

    @property
    def name(self) -> str:
        """
        Name of this filter option as valid python object name.
        """
        return f"{self.column_name}_{self.operator.suffix}"

    @property
    def alias(self) -> str:
        """
        Name of this filter option as camel-cased query option alias.
        """
        return f"{self.column_alias}[{self.operator.suffix}]"

    def input_type(self) -> Type:
        """
        Returns the valid input type for this filter option.
        """

        # assume column data type by default
        input_type = self.column.type.python_type

        if self.operator.input_type:
            # use operator input type it's explicitly specified
            input_type = self.operator.input_type

            if input_type == list:
                # in case of explicit lists, specify type of list based on column type
                input_type = List[self.column.type.python_type]

        return input_type | None

    def query_parameter(self) -> FieldInfo:
        """
        Returns the fastapi query parameter for this filter option.
        """
        return Query(
            None,
            alias=self.alias,
            title=f"Filter: `{self.alias}`",
            description=f"Filter results by `{self.column_alias}` using *{self.operator.description}* comparison.",
        )


class FilterOptions(List[FilterOption]):
    """
    Collection of `FilterOption` objects.
    This class provides a constructor to conveniently generate appropriate filter options for a given sqlalchemy data model.
    """

    def __init__(
        self,
        models: Type[DeclarativeBase] | Iterable[Type[DeclarativeBase]],
        include: str | Iterable[str] = (),
        exclude: str | Iterable[str] = (),
        include_related: Iterable[str] = (),
    ):
        # ensure models are iterable
        if not isinstance(models, Iterable):
            models = (models,)

        # add filter options from models
        for model in models:
            self.from_model(model)

        # add related filter options
        for model in models:
            self.from_relationship(model, include_related)

        # remove unwanted filter options
        self.enforce_include_exclude(include, exclude)

        # sort filter options alphabetically
        self.sort(key=lambda option: option.column_name)

    def enforce_include_exclude(
        self,
        include: str | Iterable[str] = (),
        exclude: str | Iterable[str] = (),
    ) -> Self:
        """
        Enforce include and exclude statements.

        If `include` values are set, only matching filter options will be included in the generated list.
        If `exclude` values are set, all matching filter options will be removed from the generated list.

        The include/exclude values are matched by either...
        - option name (e.g. `foo_bar_eq`)
        - option alias (e.g. `fooBar[eq]`)
        - column name (e.g. `foo_bar`)
        - column alias (e.g. `fooBar`)
        - bracketed operator suffix (e.g. `[eq]`)

        These can be mixed and matched as needed, resulting in a very flexible selection of generated filter options.

        ---

        Example:
        ```
            include = ("foo_bar", "baz[gt]", "baz[lt]")
            exclude = ("fooBar[null]", "[eq]")
        ```

        This will result in the following filter options:
        - FooBar[neq], FooBar[lt], FooBar[lte], etc...
        - *not* including FooBar[null] and FooBar[eq]
        - Baz[gt], Baz[lt]
        """

        # ensure the include/exclude arguments are iterable
        if type(include) == str:
            include = (include,)
        if type(exclude) == str:
            exclude = (exclude,)

        remove_options = set()
        for option in self:
            option_ids = (
                option.name,
                option.alias,
                option.column_name,
                option.column_alias,
                f"[{option.operator.suffix}]",
            )

            if include and not (set(option_ids) & set(include)):
                remove_options.add(option)
            if exclude and (set(option_ids) & set(exclude)):
                remove_options.add(option)

        for option in remove_options:
            self.remove(option)

        return self

    def from_column(
        self,
        column: Column,
        relationship_chain: Iterable[str] = (),
    ) -> Self:
        """
        Generate all applicable filter options for a given sqlalchemy `column`.
        """

        # ensure uniqueness, skip if this column is already represented
        if column not in [option.column for option in self]:
            # add filter option for each applicable operator
            for operator in OPERATORS:
                if operator.supports_column(column):
                    self.append(
                        FilterOption(column, operator, relationship_chain)
                    )

        return self

    def from_model(
        self,
        model: Type[DeclarativeBase],
    ) -> Self:
        """
        Generate filter options for all columns of a given sqlalchemy `model`.

        If `include` is set, only these column names will be considered.
        If `exclude` is set, these column names will not be considered.
        """
        assert (inspected_model := inspect(model))

        for column in inspected_model.columns:
            self.from_column(column)

        return self

    def from_relationship(
        self,
        model: Type[DeclarativeBase] | Type[Mapper],
        include_related: Iterable[str],
        _relationship_chain: Iterable[str] = (),
    ) -> Self:
        """
        Generate filter options for columns that belong to a relationship of am sqlalchemy model, instead of the model itself.
        The related attributes given in `include_related` must be encoded as "relationship.attribute", with a dot separating the *relationship name* and the *remote attribute name*.
        It is possible to chain relationships, as in "foo.bar.baz.attribute".
        """
        assert (inspected_model := inspect(model))

        for related_attr in include_related:
            rel_name, col_name = related_attr.split(".", maxsplit=1)

            relationship = inspected_model.relationships.get(rel_name)
            column = relationship.mapper.columns.get(col_name)

            if relationship:
                if column:
                    self.from_column(
                        column,
                        (*_relationship_chain, rel_name),
                    )
                elif "." in col_name:  # recurse
                    self.from_relationship(
                        relationship.mapper,
                        col_name,
                        (*_relationship_chain, rel_name),
                    )

        return self


class Filter(BaseModel):
    """
    Filter class, containing dynamically generated fields for query parameters.
    Can `apply()` filter statements to an sqlalchemy `Select` object.
    """

    __filter_options__: FilterOptions | list = []  # stub

    def apply(self, q: Select) -> Select:
        """
        Applies this filter to an sqlalchemy `Select` query.
        """

        for option in self.__filter_options__:
            query_parameter: QueryType = getattr(self, option.name)

            if query_parameter is not None:
                q = q.filter(
                    option.operator.compare_func(
                        option.column,
                        query_parameter,
                    )
                )

        return q

    def dict(self, *args, **kwargs) -> dict:
        kwargs.setdefault("by_alias", True)
        kwargs.setdefault("exclude_none", True)
        return super().dict(*args, **kwargs)

    @classmethod
    def create(
        cls,
        models: Type[DeclarativeBase] | Iterable[Type[DeclarativeBase]],
        include: str | Iterable[str] = (),
        exclude: str | Iterable[str] = (),
        include_related: Iterable[str] = (),
    ) -> Type[Filter]:
        """
        Dynamically creates a filter class with all filter options as fastapi `Query` parameters.

        The generated class contains all filter options as fastapi `Query` objects.
        These are annotated with openapi metadata (description, title, alias) which shows up properly in openapi documentation.
        The class inherits from pydantic's `BaseModel`, which will autogenerate an `__init__` method, whihc is required for fastapi to recognize the query parameters.
        This also provides the ability to return the filter object directly as part of the response, if you want to return the filters the client has input.
        """

        # generate filter options
        filter_options: List[FilterOption] = FilterOptions(
            models, include, exclude, include_related
        )

        # set required attributes
        attributes = {
            "__filter_options__": filter_options,
            "__annotations__": {},
        }

        # compile filter options as pydantic fields
        for option in filter_options:
            attributes[option.name] = Field(option.query_parameter())
            attributes["__annotations__"][option.name] = option.input_type()

        # dynamically create filter class
        return type(
            "Filterinator",
            (cls,),
            attributes,
        )


def get_filter(
    models: Type[DeclarativeBase] | Iterable[Type[DeclarativeBase]],
    include: str | Iterable[str] = (),
    exclude: str | Iterable[str] = (),
    include_related: Iterable[str] = (),
) -> Type[Filter]:
    """
    Dependency function to dynamically generate a query parameter class for filter options based on the given sqlalchemy model(s) and additional parameters.

    If `include` values are set, only these values are included in the output.
    If `exclude` values are set, they will be excluded from the returned output.

    Returns a `Filter` object which holds all filter options passed through the API.
    You can apply the filters to an sqlalchemy `Select` object using the `Filter.apply` method.
    """

    return Filter.create(models, include, exclude, include_related)
