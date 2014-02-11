"""
Created on Feb 10, 2014

@author: Vadim Markovtsev <v.markovtsev@samsung.com>
"""


import logging
import multiprocessing as mp
import queue
import threading
import tornado.escape
import tornado.ioloop as ioloop
import tornado.web as web
import uuid

import config


class ServiceHandler(web.RequestHandler):
    def initialize(self, server):
        self.server = server

    @tornado.web.asynchronous
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        self.server.send_command(self, data)

    def finish_post(self, result):
        self.finish(result)


class WebStatusServer:
    def __init__(self, cmd_queue_in, cmd_queue_out):
        self.application = web.Application([
            (r"/service", ServiceHandler, {"server": self}),
            (r"/(js/.*)", web.StaticFileHandler, {'path':
                                                  config.web_status_root}),
            (r"/(css/.*)", web.StaticFileHandler, {'path':
                                                   config.web_status_root}),
            (r"/(fonts/.*)", web.StaticFileHandler, {'path':
                                                     config.web_status_root}),
            (r"/(.+\.html)", web.StaticFileHandler, {'path':
                                                     config.web_status_root}),
            (r"/(veles.png)", web.StaticFileHandler, {'path':
                                                      config.web_status_root}),
            (r"/", web.RedirectHandler, {"url": "/status.html",
                                        "permanent": True}),
            (r"", web.RedirectHandler, {"url": "/status.html",
                                       "permanent": True})
        ])
        self.application.listen(config.web_status_port)
        self.cmd_queue_in = cmd_queue_in
        self.cmd_queue_out = cmd_queue_out
        self.cmd_thread = threading.Thread(target=self.cmd_loop)
        self.pending_requests = {}

    def send_command(self, handler, cmd):
        request_id = uuid.uuid4()
        self.pending_requests[request_id] = handler
        self.cmd_queue_out.put_nowait({"request": request_id, "body": cmd})

    def cmd_loop(self):
        while True:
            cmd = self.cmd_queue_in.get()
            if cmd == "exit":
                ioloop.IOLoop.instance().stop()
                break
            request_id = cmd["request"]
            self.pending_requests[request_id].finish_post(cmd["result"])
            del(self.pending_requests[request_id])

    def run(self):
        self.cmd_thread.start()
        ioloop.IOLoop.instance().start()
        self.cmd_thread.join()


class WebStatus:
    """
    Operates a web server based on Tornado to show various runtime information.

    TODO(v.markovtsev): when twisted.web.static is ported, we should rewrite.
    See:
    http://twistedmatrix.com/trac/ticket/6177
    https://twistedmatrix.com/documents/13.2.0/web/howto/using-twistedweb.html
    http://twistedmatrix.com/documents/current/web/howto/web-in-60/index.html
    """

    @staticmethod
    def start_web_server(cmd_queue_in, cmd_queue_out):
        WebStatusServer(cmd_queue_in, cmd_queue_out).run()

    def __init__(self, master_server):
        super(WebStatus, self).__init__()
        self.exiting = False
        self.master_server = master_server
        self.cmd_queue_in = mp.Queue()
        self.cmd_queue_out = mp.Queue()
        self.cmd_thread = threading.Thread(target=self.cmd_loop)
        self.process = mp.Process(target=WebStatus.start_web_server,
                                  args=(self.cmd_queue_out, self.cmd_queue_in))

    def run(self):
        self.cmd_thread.start()
        self.process.start()

    def stop(self):
        self.exiting = True
        self.cmd_queue_out.put("exit")
        self.cmd_thread.join()

    def cmd_loop(self):
        while not self.exiting:
            try:
                while True:
                    cmd = self.cmd_queue_in.get_nowait()
                    ret = self.master_server.fulfill_request(cmd["body"])
                    self.cmd_queue_out.put_nowait({"request": cmd["request"],
                                                   "result": ret})
            except queue.Empty:
                pass


class Dummy:
    def fulfill_request(self, data):
        print(str(data))
        return "AJAX WORKS!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ws = WebStatus(Dummy())
    ws.run()
