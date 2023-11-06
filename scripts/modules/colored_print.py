#! /usr/bin/env python
# -*- coding: utf-8 -*-

class Color:
    RED       = '\033[91m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    BLUE      = '\033[94m'
    PURPLE    = '\033[95m'
    CYAN      = '\033[96m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'
       
def printr(str):
  print(Color.RED + "{}".format(str) + Color.END)

def printg(str):
  print(Color.GREEN + "{}".format(str) + Color.END)

def printy(str):
  print(Color.YELLOW + "{}".format(str) + Color.END)

def printb(str):
  print(Color.BLUE + "{}".format(str) + Color.END)

def printp(str):
  print(Color.PURPLE + "{}".format(str) + Color.END)

def printc(str):
  print(Color.CYAN + "{}".format(str) + Color.END)