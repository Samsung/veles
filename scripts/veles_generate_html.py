#!/usr/bin/python3
# encoding: utf-8
'''
This scripts generates an HTML page with all velescli's command line arguments,
 allowing for fast command line composition

'''
import argparse
import os
import sys
from scripts.velescli import Main
from veles.config import root

WEB_FOLDER = root.common.web_folder


def main():
    parser = Main.init_parser()
    arguments = parser._actions
    path_to_file = os.path.join(WEB_FOLDER, "frontend_template.html")
    path_to_out = os.path.join(WEB_FOLDER, "frontend.html")
    list_lines = []
    for tuple_obj in sorted([convert_argument(arg) for arg in arguments]):
        list_lines.append(tuple_obj[1])
    html = ''.join(list_lines)
    with open(path_to_file, "r") as fin:
        sin = fin.read()
        str_rp = ("            <!-- INSERT ARGUMENTS HERE-->\n")
        sout = sin.replace(str_rp, html)

    with open(path_to_out, "w") as fout:
        fout.write(sout)


def convert_argument(arg):
    choices = arg.choices
    required = arg.required
    arg_line = ""
    if choices is not None:
        arg_line = convert_choices(arg)
    else:
        if isinstance(arg, argparse._StoreTrueAction):
            arg_line = convert_boolean(arg)
        if isinstance(arg, argparse._StoreAction):
            arg_line = convert_string(arg)
    imp = 0 if required else 1
    return (imp, arg_line)


def convert_string(arg):
    dest = arg.dest
    hlp = arg.help
    required = arg.required
    importance = 'Obligatory' if required else 'Optional'
    importance_class = 'danger' if required else 'default'
    arg_line = ("""
            <div class="panel panel-primary argument">
                <div class="panel-heading">
                  <span class="label label-%s argtype">%s</span>
                  <h3 class="panel-title">%s</h3>
                </div>
                <div class="panel-body">
                    <div class="pull-right description">
                      <p>%s.</p>
                    </div>
                    <div class="input-group">
                      <span class="input-group-addon">%s</span>
                      <input type="text" class="form-control" placeholder="">
                    </div>
                </div>
            </div>
                """ % (importance_class, importance, dest, hlp, dest))
    return arg_line


def convert_boolean(arg):
    dest = arg.dest
    hlp = arg.help
    required = arg.required
    importance = 'Obligatory' if required else 'Optional'
    importance_class = 'danger' if required else 'default'
    arg_line = ("""
            <div class="panel panel-primary argument">
                <div class="panel-heading">
                  <span class="label label-%s argtype">%s</span>
                  <h3 class="panel-title">%s</h3>
                </div>
                <div class="panel-body">
                    <div class="pull-right description">
                      <p>%s.</p>
                    </div>
                    <div class="bootstrap-switch-container">
                      <input type="checkbox" class="switch" data-on-text="Yes"
                       data-off-text="No" data-size="large" checked />
                    </div>
                </div>
            </div>
""" % (importance_class, importance, str(dest), hlp))
    return arg_line


def convert_choices(arg):
    choices = arg.choices
    dest = arg.dest
    hlp = arg.help
    required = arg.required
    importance = 'Obligatory' if required else 'Optional'
    importance_class = 'danger' if required else 'default'
    choices_lines = ''
    for choice in choices:
        line_ch = ("""
                        <li role="presentation"><a role="menuitem"tabindex="-1"
                        href="#">%s</a></li>
""" % (str(choice)))
        choices_lines += line_ch
        arg_line = ("""
            <div class="panel panel-primary argument">
                <div class="panel-heading">
                  <span class="label label-%s argtype">%s</span>
                  <h3 class="panel-title">%s</h3>
                </div>
                <div class="panel-body">
                    <div class="pull-right description">
                      <p>%s.</p>
                    </div>
                    <div class="dropdown">
                      <button class="btn btn-default dropdown-toggle"
                      type="button" id="dropdownMenu1" data-toggle="dropdown">
                        %s
                        <span class="caret"></span>
                      </button>
                      <ul class="dropdown-menu" role="menu"
                      aria-labelledby="dropdownMenu1">
                        %s
                      </ul>
                    </div>
                </div>
            </div>
""" % (importance_class, importance, dest, hlp,
       str(list(choices)[0]), choices_lines))
    return arg_line

if __name__ == "__main__":
    retcode = main()
    sys.exit(retcode)
