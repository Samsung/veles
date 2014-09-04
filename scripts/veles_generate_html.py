#!/usr/bin/python3
# encoding: utf-8
'''
Generate html file for argumets of veles command line.

'''
import os
import sys
from scripts.velescli import Main


class Generate_html():
    def run(self):
        parser = Main.init_parser()
        line_for_action = ''
        actions = parser.__dict__["_actions"]
        path_to_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "web_status/frontend.html")
        path_to_out = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "web_status/generate_velescli.html")
        line_to_write = ''
        for action in actions:
            choices = action.choices
            dest = action.dest
            hlp = action.help
            required = action.required
            choices_lines = ''
            importance = 'Obligatory' if required is True else 'Not obligatory'
            if choices is not None:
                for choice in choices:
                    line_ch = ('                        <li role="presentation'
                               + '"><a role="menuitem" tabindex="-1" href="#">'
                               + str(choice) + '</a></li>\n')
                    choices_lines += line_ch
                line_for_action = ('            <div class="panel panel-primar'
                                   + 'y argument">\n                <div class'
                                   + '="panel-heading">\n                  <sp'
                                   + 'an'
                                   + ' class="label label-danger argtype">' +
                                   importance + '</span>\n                  <h'
                                   + '3 class="panel-title">' + str(dest) +
                                   '</h3>\n                </div>\n           '
                                   + '     <div class'
                                   + '="panel-body">\n                    <di'
                                   + 'v class="pull-right description">\n    '
                                   + '                  <p>' + hlp + '.</p>\n'
                                   + '                    </div>\n            '
                                   + '        <div class="dropdown">\n        '
                                   + '              <button class="btn btn-def'
                                   + 'ault dropdown-toggle" type="button" id="'
                                   + 'dropdownMenu1" data-toggle="dropdown">\n'
                                   + '                        ' +
                                   str(list(choices)[0]) + '\n                '
                                   + '      '
                                   + '  <span class="caret"></span>\n         '
                                   + '             </button>\n                '
                                   + '      <ul class="dropdown-menu" role="me'
                                   + 'nu" aria-labelledby="dropdownMenu1">\n'
                                   + choices_lines + '                      </'
                                   + 'ul>\n                    </div>\n       '
                                   + '         </div>\n            </div>\n')
            else:
                line_for_action = ('            <div class="panel panel-primar'
                                   + 'y argument">\n                <div class'
                                   + '="panel-heading">\n                  <sp'
                                   + 'an'
                                   + ' class="label label-danger argtype">' +
                                   importance + '</span>\n                  <h'
                                   + '3 class="panel-title">' + str(dest) +
                                   '</h3>\n                </div>\n           '
                                   + '     <div class'
                                   + '="panel-body">\n                    <di'
                                   + 'v class="pull-right description">\n    '
                                   + '                  <p>' + hlp + '.</p>\n'
                                   + '                    </div>\n            '
                                   + '        <div class="input-group">\n  <sp'
                                   + 'an class="input-group-addon">' + dest +
                                   '</span>\n  <input type="text" class="form-'
                                   + 'control" placeholder="">\n</div>'
                                   + '                </div>\n            </di'
                                   + 'v>\n')
            if line_for_action is not '':
                if required is not True:
                    line_to_write += line_for_action
                else:
                    line_to_write = line_for_action + line_to_write
        with open(path_to_file, "r") as fin:
            sin = fin.read()
            str_rp = ('            <!-- INSERT ARGUMENTS HERE-->\n' +
                      '            <div class="panel panel-primary argument"' +
                      '>\n                <div class="panel-heading">\n       '
                      + '           <span class="label label-danger argtype">O'
                      + 'bligatory</span>\n                  <h3 class="panel-'
                      + 'title">Mode</h3>                  \n                <'
                      + '/div>\n                <div class="panel-body">   \n '
                      + '                   <d'
                      + 'iv class="pull-right description">\n                 '
                      + '     <p>Selects VELES operation mode. Standalone is t'
                      + 'he best choice for debugging workflows and relatively'
                      + ' small tasks. Master mode runs VELES workflow server '
                      + 'which accepts slaves. Slave mode is the headless VELE'
                      + 'S instance which requests jobs from a master and retu'
                      + 'rns results.</p>\n                    ' +
                      '</div>                 \n                    <d'
                      + 'iv class="dropdown">\n                      <button c'
                      + 'lass="btn btn-default dropdown-toggle" type="button" '
                      + 'id="dropdownMenu1" data-toggle="dropdown">\n         '
                      + '               Standalone\n                        <s'
                      + 'pan class="caret"></span>\n                      </bu'
                      + 'tton>\n                      <ul class="dropdown-men'
                      + 'u" role="menu" aria-labelledby="dropdownMenu1">\n    '
                      + '                    <li role="presentation"><a role="'
                      + 'menuitem" tabindex="-1" href="#">Standalone</a></li>'
                      + '\n                        <li role="presentation"><a'
                      + ' role="menuitem" tabindex="-1" href="#">Master</a></l'
                      + 'i>\n                        <li role="presentation"><'
                      + 'a role="menuitem" tabindex="-1" href="#">Slave</a></l'
                      + 'i>\n                      </ul>\n                    '
                      + '</div>                    \n                </div>\n '
                      + '           </div>\n')
            sout = sin.replace(str_rp, str_rp +
                               line_to_write)
        with open(path_to_out, "w") as fout:
            fout.write(sout)

if __name__ == "__main__":
    retcode = Generate_html().run()
    sys.exit(retcode)
