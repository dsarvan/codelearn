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
# emp1.employeeName() we don't need to pass in self

# but when we call the method on the class we do have
# to pass in the instance and that get pass in as self

# when we run empl.employeeName() it get transformed into
# Employee.employeeName() and passes in emp1 as self

"""
Class variables are variables that are shared among all instances of a class.
So while instance variables can be unique for each instance like name, email,
and pay. Class variables should be the same for each instance.

"""

class Employee:

	def __init__(self, fname, lname, stock):
		self.fname = fname
		self.lname = lname
		self.email = fname + lname + '@mail.com'
		self.stock = stock

	def employeeName(self):
		return f"{self.fname} {self.lname}"

	def applyRaise(self):
		self.stock = int(self.stock * 1.04)


emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

print(emp1.stock)
emp1.applyRaise()
print(emp1.stock)


# When we access the class variables we need to either access
# them through class itself or an instance of the class

class Employee:

	raise_amount = 1.04

	def __init__(self, fname, lname, stock):
		self.fname = fname
		self.lname = lname
		self.email = fname + lname + '@mail.com'
		self.stock = stock

	def employeeName(self):
		return f"{self.fname} {self.lname}"

	def applyRaise(self):
		# access through class itself
		#self.stock = int(self.stock * Employee.raise_amount)
		# access through instance of the class
		self.stock = int(self.stock * self.raise_amount)


emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

# access the class variable from both through the
# class itself as well as an instance of the class
print(emp1.raise_amount)
print(emp2.raise_amount)
print(Employee.raise_amount)

# when we try to access an attribute of an instance it will first
# check if the instance contains that attribute and if it doesn't
# then it will see if the class or any class that it inherit from
# contains that attribute

# so when we access raise_amount from our instances here they don't
# actually have the attributes themselves they accessing the class's
# raise_amount attribute

print(emp1.stock)
emp1.applyRaise()
print(emp1.stock)

# print out the namespace of class Employee
print(Employee.__dict__) # contains raise_amount = 1.04

# print out the namespace of instances emp1 & emp2
print(emp1.__dict__)
print(emp2.__dict__)

# change the raise_amount for the class and all of the instances
Employee.raise_amount = 1.05
print(emp1.raise_amount)
print(emp2.raise_amount)
print(Employee.raise_amount)

# change the raise_amount using an instance emp1
emp1.raise_amount = 1.05
print(emp1.raise_amount) # only change raise_amount for emp1
print(emp2.raise_amount)
print(Employee.raise_amount)

# change the raise_amount using an instance emp2
emp2.raise_amount = 1.05
print(emp1.raise_amount)
print(emp2.raise_amount) # only change raise_amount for emp2
print(Employee.raise_amount)

# changing the raise_amount using an instance will change the raise_amount only
# for that instance

# when we made this assignment it actually created the raise_amount attribute
# within emp1 and emp2

# now again print out the namespace of class Employee and instances emp1 & emp2
print(Employee.__dict__) # contains raise_amount = 1.05
print(emp1.__dict__)	 # contains raise_amount = 1.05
print(emp2.__dict__)	 # contains raise_amount = 1.05

# prefer self.raise_amount than Employee.raise_amount because that will give the
# ability to change that amount for a single instance if we really want it to
# and also using self here will allow any sub class to overwrite that constant
# if they want it to

# example of class variable where using self not make sense

class Employee:

	employee_num = 0
	raise_amount = 1.04

	def __init__(self, fname, lname, stock):
		self.fname = fname
		self.lname = lname
		self.email = fname + lname + '@mail.com'
		self.stock = stock

		Employee.employee_num += 1

	def employeeName(self):
		return f"{self.fname} {self.lname}"

	def applyRaise(self):
		self.stock = int(self.stock * self.raise_amount)

# since __init__ method runs every time we create new employee
# use Employee.employee_num += 1 instead of self.employee_num += 1
# because with raises it nice to have that constant class value that
# can be overwritten per instance if we really need it to be but in
# this case there is no use case we could think of where we would want
# total number of employees to be different for any one instances

print(Employee.employee_num)

emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

print(Employee.employee_num)

"""
regular methods, class methods and static methods:

* regular methods in a class automatically take the instance as the first
  argument and by convention we call this 'self'

* class methods take the class as the first argument and by convention we call
  this 'cls'

* to turn a regular method into a class method add @classmethod decorator to a
  regular method

* decorator alters the functionality of a method

"""

class Employee:

	employee_num = 0
	raise_amount = 1.04

	def __init__(self, fname, lname, stock):
		self.fname = fname
		self.lname = lname
		self.email = fname + lname + '@mail.com'
		self.stock = stock

		Employee.employee_num += 1

	def employeeName(self):
		return f"{self.fname} {self.lname}"

	def applyRaise(self):
		self.stock = int(self.stock * self.raise_amount)

	@classmethod
	def setRaiseAmount(cls, amount):
		cls.raise_amount = amount


emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

print(emp1.raise_amount)
print(emp2.raise_amount)
print(Employee.raise_amount)

# change raise_amount from 4% to 5% by
# using setRaiseAmount() class method
Employee.setRaiseAmount(1.05)

print(emp1.raise_amount)
print(emp2.raise_amount)
print(Employee.raise_amount)

# running Employee.setRaiseAmount(1.05) is equivalent
# to setting Employee.raise_amount = 1.05

# we can run class methods from instances as well but that doesn't make lot of
# sense and we don't people doing it but to show what that look like

# run the class method using the instance
emp1.setRaiseAmount(1.06)

print(emp1.raise_amount)
print(emp2.raise_amount)
print(Employee.raise_amount)

# running the class method from the instance still changes the class variable
# and sets all of the class variable and both instances amount to that 6% that
# we passed in

# we can use these class methods in order to provide multiple ways of creating
# our instances

# example of using class methods as alternative constructors

emp1str = 'Edward-Teller-4000'
emp2str = 'Corey-Schafer-5000'
emp3str = 'Richard-Fermi-7000'

fname, lname, stock = emp1str.split('-')

emp1 = Employee(fname, lname, stock)

print(emp1.email)
print(emp1.stock)

# if above is the common use case of how someone using this class than we don't
# want them to parse these strings every time they want to create a new employee
# so let just create an alternative constructor that allows them to pass in the
# string and then we create the employee for them

# use 'from' by convention for the method using as an alternative constructor

class Employee:

	employee_num = 0
	raise_amount = 1.04

	def __init__(self, fname, lname, stock):
		self.fname = fname
		self.lname = lname
		self.email = fname + lname + '@mail.com'
		self.stock = stock

		Employee.employee_num += 1

	def employeeName(self):
		return f"{self.fname} {self.lname}"

	def applyRaise(self):
		self.stock = int(self.stock * self.raise_amount)

	@classmethod
	def setRaiseAmount(cls, amount):
		cls.raise_amount = amount

	@classmethod # use this method as an alternative constructor
	def fromString(cls, empstr):
		fname, lname, stock = empstr.split('-')
		return cls(fname, lname, stock)


emp1str = 'Edward-Teller-4000'
emp2str = 'Corey-Schafer-5000'
emp3str = 'Richard-Fermi-7000'

emp1 = Employee.fromString(emp1str)
emp2 = Employee.fromString(emp2str)
emp3 = Employee.fromString(emp3str)

print(emp1.email)
print(emp1.stock)

# Note: look at datetime module in python to better
#       understand an alternative constructor


"""
static method:

* regular methods automatically pass the instance as the first argument and we
  call that 'self' by convention

* class methods automatically pass the class as the first argument and we call
  that 'cls' by convention

* static methods don't pass anything automatically (instance/class)

* static methods behave just like regular functions except we include them in
  classes because they have some logical connection with the class

"""

# sometimes people write regular methods or class methods that actually should
# be static methods and usually a give away that the method should be static
# method is if you don't access the instance or the class anywhere within the
# function

# for example consider a simple function that would take in a date and return
# whether or not that was a work day, so that has a logical connection to our
# Employee class but it doesn't actually depend on any specific instance or
# class variable

# in python, dates have these week day methods where Monday == 0 and Sunday == 6


class Employee:

	employee_num = 0
	raise_amount = 1.04

	def __init__(self, fname, lname, stock):
		self.fname = fname
		self.lname = lname
		self.email = fname + lname + '@mail.com'
		self.stock = stock

		Employee.employee_num += 1

	def employeeName(self):
		return f"{self.fname} {self.lname}"

	def applyRaise(self):
		self.stock = int(self.stock * self.raise_amount)

	@classmethod
	def setRaiseAmount(cls, amount):
		cls.raise_amount = amount

	@classmethod # use this method as an alternative constructor
	def fromString(cls, empstr):
		fname, lname, stock = empstr.split('-')
		return cls(fname, lname, stock)

	@staticmethod
	def isWorkday(day):
		"""function to return the day is work day or not"""
		return False if day.weekday() in (5, 6) else True


emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

import datetime
date = datetime.date(2023, 4, 30)

print(Employee.isWorkday(date))

"""
Summary:
* we learn the difference between regular instance methods, class methods which
  can also be used as an alternative constructors and static methods which don't
  operate on the instance or the class

"""
