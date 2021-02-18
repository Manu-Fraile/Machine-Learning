# -*- coding: utf-8 -*-

import monkdata as m
import dtree as d
t = d.buildTree(m.monk1, m.attributes)
print(d.check(t, m.monk1test))