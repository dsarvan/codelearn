#!/usr/bin/env python
# File: pyobject.py
# Name: D.Saravanan
# Date: 29/04/2023

""" Script for object-oriented programming """

"""
creating and instantiating a class in python:

* class allow us to logical group data and functions in a way 
  that easy to reuse and also easy to built upon if need be 

* data and functions that are associated with a specific class 
  we call those attributes and methods

* attributes: a data that is associated with a class

* methods: a function that is associated with a class

* a class is basically a blue print for creating instances and 
  each unique instance that we create with this class will be an 
  instance of that class

* instance variables contain data that is unique to each instance

"""

class Employee:
	pass


emp1 = Employee()
emp2 = Employee()

# emp1 and emp2 are Employee objects and both have different locations in memory
print(emp1)
print(emp2)

"""
* manually create instance variables for each Employee
* each of these instances have attributes that are unique to them

"""
emp1.fname = "Edward"
emp1.lname = "Teller"
emp1.email = "edwardteller@mail.com"
emp1.stock = 4000

emp2.fname = "Corey"
emp2.lname = "Schafer"
emp2.email = "coreyschafer@mail.com"
emp2.stock = 5000

print(emp1.email)
print(emp2.email)

# display the full name of an Employee
print(f"{emp1.fname} {emp1.lname}")
print(f"{emp2.fname} {emp2.lname}")

# set all of these information for each Employee when 
# they are created rather than doing all of this manually
# with special __init__ method (constructor)

# when we create methods within a class they receive the instance as the first
# argument automatically and by convention we should call the instance 'self'

class Employee:
	
	def __init__(self, fname, lname, stock):
		self.fname = fname
		self.lname = lname
		self.email = fname + lname + '@mail.com'
		self.stock = stock

	# each method within a class automatically
	# takes the instance as the first argument
	def employeeName(self):
		return f"{self.fname} {self.lname}"


# when we create Employee the instance is passed 
# automatically so we can leave off self

# emp1 and emp2 are passed in as self and then 
# it will set all of these attributes 

# fname, lname, stock are all attributes of the class Employee

emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

print(emp1.email)
print(emp2.email)

print(emp1.employeeName())
print(emp2.employeeName())

# we can also run these methods using the class name itself by 
# manually pass in the instance as an argument
print(Employee.employeeName(emp1))
print(Employee.employeeName(emp2))

# emp1 which is an instance then call the method with 
# emp1.employeeName() we  don't need to pass in self 

# but when we call the method on the class we do have 
# to pass in the instance and that get pass in as self

# when we run empl.employeeName() it get transformed into 
# Employee.employeeName() and passes in emp1 as self
