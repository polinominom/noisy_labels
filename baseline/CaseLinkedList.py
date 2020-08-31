import numpy as np

class Node:
  def __init__(self, value, info):
    self.value = value
    self.info = info
    self.next = None

class CaseLinkedList:
  def __init__(self):
    self.head = None

  def add(self, value, info):
    n = Node(value, info)
    if self.head == None:
      self.head = n
      return
    
    if n.value > self.head.value:
      a = self.head
      self.head = n
      self.head.next = a
      return

    if self.head.next == None:
      if n.value <= self.head.value:
        self.head.next = n
      return

    a = self.head
    while a.next != None:
      t = a.next
      if n.value > t.value:
        a.next = n
        n.next = t
        return
      a = a.next
    
    a.next = n

  def get_values(self):
    result = []
    a = self.head
    while a != None:
      result.append(a.value)
      a = a.next

    return np.array(result)

  def get_info(self):
    result = []
    a = self.head
    while a != None:
      result.append(a.info)
      a = a.next

    return np.array(result)