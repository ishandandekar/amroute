import sys

import osmium

src = sys.argv[1]
dst = sys.argv[2]


class PassThrough(osmium.SimpleHandler):
    def node(self, n):
        self.writer.add_node(n)

    def way(self, w):
        self.writer.add_way(w)

    def relation(self, r):
        self.writer.add_relation(r)


h = PassThrough()
h.writer = osmium.SimpleWriter(dst)
osmium.apply(src, h)
h.writer.close()
print(f"Done: {dst}")
