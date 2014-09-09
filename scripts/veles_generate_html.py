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
        str_rp = ("<!-- INSERT ARGUMENTS HERE-->")
        sout = sin.replace(str_rp, html)

    with open(path_to_out, "w") as fout:
        fout.write(sout)


def convert_argument(arg):
    choices = arg.choices
    required = arg.required
    dest = arg.dest
    hlp = arg.help
    arg_mode = getattr(arg, "mode", ["standalone", "master", "slave"])
    arg_line = ""
    if choices is not None:
        arg_line = convert_choices(arg, arg_mode)
    else:
        if isinstance(arg, argparse._StoreTrueAction):
            arg_line = convert_boolean(arg, arg_mode)
        if isinstance(arg, argparse._StoreAction):
            arg_line = convert_string(arg, arg_mode)
    imp = (int(required)) ^ 1

    importance = 'Obligatory' if required else 'Optional'
    importance_class = 'danger' if required else 'default'
    template_line = """
            <div class="panel panel-primary argument %s">
                <div class="panel-heading %s">
                  <span class="label label-%s argtype">%s</span>
                  <h3 class="panel-title">%s</h3>
                </div>
                <div class="panel-body">
                    <div class="pull-right description">
                      <p>%s.</p>
                    </div>
                    %s
                </div>
            </div>""" % (" ".join(arg_mode), " ".join(arg_mode),
                         importance_class, importance, dest, hlp, arg_line)
    return (imp, template_line)


def convert_string(arg, arg_mode):
    dest = arg.dest
    default = arg.default
    arg_line = ("""
                    <div class="input-group">
                     <span class="input-group-addon">%s</span>
                     <input type="text" class="form-control %s" placeholder=%s>
                    </div>""" % (dest, " ".join(arg_mode), default))
    return arg_line


def convert_boolean(arg, arg_mode):
    default = arg.default
    checked = "checked" if default else ""
    arg_line = ("""
                    <div class="bootstrap-switch-container">
                      <input type="checkbox" class="switch %s"
                       data-on-text="Yes"
                       data-off-text="No" data-size="large" %s />
                    </div>""" % (" ".join(arg_mode), checked))
    return arg_line


def convert_choices(arg, arg_mode):
    choices = arg.choices
    choices_lines = ''
    default = arg.default
    for choice in choices:
        line_ch = ("""
                        <li role="presentation"><a role=
                   "menuitem"tabindex="-1"href="#">%s</a></li>""" % choice)
        choices_lines += line_ch
        arg_line = ("""
                    <div class="dropdown">
                      <button class="btn btn-default dropdown-toggle %s"
                      type="button" id="dropdownMenu1" data-toggle="dropdown">
                        %s
                        <span class="caret"></span>
                      </button>
                      <ul class="dropdown-menu" role="menu"
                      aria-labelledby="dropdownMenu1">
                        %s
                      </ul>
                    </div>""" % (" ".join(arg_mode), default, choices_lines))
    return arg_line

if __name__ == "__main__":
    retcode = main()
    sys.exit(retcode)
