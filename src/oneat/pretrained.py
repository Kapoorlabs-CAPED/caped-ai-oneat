"""
Created on Tue Dec 21 16:54:50 2021

@author: stardist devs
"""

# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
from warnings import warn

from csbdeep.utils import _raise
from csbdeep.utils.six import Path
from csbdeep.utils.tf import keras_import

from oneat.NEATUtils.utils import load_json

get_file = keras_import("utils", "get_file")


_MODELS = {}
_ALIASES = {}


def clear_models_and_aliases(*cls):
    if len(cls) == 0:
        _MODELS.clear()
        _ALIASES.clear()
    else:
        for c in cls:
            if c in _MODELS:
                del _MODELS[c]
            if c in _ALIASES:
                del _ALIASES[c]


def register_model(
    cls,
    key,
    url,
    hash,
    cordkey,
    cordurl,
    cordhash,
    catkey,
    caturl,
    cathash,
    paramkey,
    paramurl,
    paramhash,
):
    # key must be a valid file/folder name in the file system
    models = _MODELS.setdefault(cls, OrderedDict())
    key not in models or warn(
        "re-registering model '{}' (was already registered for '{}')".format(
            key, cls.__name__
        )
    )
    models[key] = dict(
        url=url,
        hash=hash,
        cordkey=cordkey,
        cordurl=cordurl,
        cordhash=cordhash,
        catkey=catkey,
        caturl=caturl,
        cathash=cathash,
        paramkey=paramkey,
        paramurl=paramurl,
        paramhash=paramhash,
    )


def register_aliases(cls, key, *names):
    # aliases can be arbitrary strings
    if len(names) == 0:
        return
    models = _MODELS.get(cls, {})
    key in models or _raise(
        ValueError(f"model '{key}' is not registered for '{cls.__name__}'")
    )
    aliases = _ALIASES.setdefault(cls, OrderedDict())
    for name in names:
        aliases.get(name, key) == key or warn(
            "alias '{}' was previously registered with model '{}' for '{}'".format(
                name, aliases[name], cls.__name__
            )
        )
        aliases[name] = key


def get_registered_models(cls, return_aliases=True, verbose=False):
    models = _MODELS.get(cls, {})
    aliases = _ALIASES.get(cls, {})
    model_keys = tuple(models.keys())
    model_aliases = {
        key: tuple(name for name in aliases if aliases[name] == key)
        for key in models
    }
    if verbose:
        # this code is very messy and should be refactored...
        _n = len(models)
        _str_model = "model" if _n == 1 else "models"
        _str_is_are = "is" if _n == 1 else "are"
        _str_colon = ":" if _n > 0 else ""
        print(
            "There {is_are} {n} registered {model_s} for '{clazz}'{c}".format(
                n=_n,
                clazz=cls.__name__,
                is_are=_str_is_are,
                model_s=_str_model,
                c=_str_colon,
            )
        )
        if _n > 0:
            print()
            _maxkeylen = 2 + max(len(key) for key in models)
            print("Name{s}Alias(es)".format(s=" " * (_maxkeylen - 4 + 3)))
            print("────{s}─────────".format(s=" " * (_maxkeylen - 4 + 3)))
            for key in models:
                _aliases = "   "
                _m = len(model_aliases[key])
                if _m > 0:
                    _aliases += "'%s'" % "', '".join(model_aliases[key])
                else:
                    _aliases += "None"
                _key = ("{s:%d}" % _maxkeylen).format(s="'%s'" % key)
                print(f"{_key}{_aliases}")
    return (model_keys, model_aliases) if return_aliases else model_keys


def get_model_details(cls, key_or_alias, verbose=False):
    models = _MODELS.get(cls, {})
    if key_or_alias in models:
        key = key_or_alias
        alias = None
    else:
        aliases = _ALIASES.get(cls, {})
        alias = key_or_alias
        alias in aliases or _raise(
            ValueError(
                "'{}' is neither a key or alias for '{}'".format(
                    alias, cls.__name__
                )
            )
        )
        key = aliases[alias]
    if verbose:
        print(
            "Found model '{model}'{alias_str} for '{clazz}'.".format(
                model=key,
                clazz=cls.__name__,
                alias_str=(
                    "" if alias is None else " with alias '%s'" % alias
                ),
            )
        )
    return key, alias, models[key]


def get_model_folder(cls, key_or_alias, target):
    key, alias, m = get_model_details(cls, key_or_alias)

    model_name = Path(
        get_file(
            fname=key + ".h5",
            origin=m["url"],
            file_hash=m["hash"],
            cache_subdir=target,
            extract=True,
        )
    )
    cord = load_json(
        get_file(
            fname=m["cordkey"] + ".json",
            origin=m["cordurl"],
            file_hash=m["cordhash"],
            cache_subdir=target,
            extract=True,
        )
    )
    cat = load_json(
        get_file(
            fname=m["catkey"] + ".json",
            origin=m["caturl"],
            file_hash=m["cathash"],
            cache_subdir=target,
            extract=True,
        )
    )
    param = load_json(
        get_file(
            fname=m["paramkey"] + ".json",
            origin=m["paramurl"],
            file_hash=m["paramhash"],
            cache_subdir=target,
            extract=True,
        )
    )

    return model_name.stem, cord, cat, param


def get_model_instance(cls, key_or_alias, target):
    model_name, cord, cat, param = get_model_folder(cls, key_or_alias, target)

    model = cls(
        config=None,
        model_dir=target,
        model_name=os.path.splitext(model_name)[0],
        catconfig=cat,
        cordconfig=cord,
    )

    return model
