"""
Created on Nov 10, 2014

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


from pkg_resources import Requirement


REQUIRED_MANIFEST_FIELDS = {
    "name", "workflow", "configuration", "short_description",
    "long_description", "author", "requires"
}


def validate_requires(requires):
    if not isinstance(requires, list):
        raise TypeError("\"requires\" must be an instance of []")
    packages = set()
    for item in requires:
        if not isinstance(item, str):
            raise TypeError("Each item in \"requires\" must be "
                            "a requirements.txt style string")
        pn = Requirement.parse(item).project_name
        if pn in packages:
            raise ValueError("Package %s was listed in \"requires\" more than "
                             "once" % pn)
        packages.add(pn)
