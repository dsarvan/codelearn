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

"""
inheritance and creating subclasses:

* inheritance allows us to inherit attributes and methods from a parent class

* if we create subclasses and get all the functionality of our parent class and
  then we can overwrite or add completely new functionality without affecting
  the parent class in any way

"""

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
		self.stock = int(self.stock * self.raise_amount)


emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)


# let create different types of employees, for example: developers and managers
# these will be good candidates for subclasses because both developers and managers
# have fname, lname, email and stock since these are already present in Employee
# class so instead of copying all these code in Developer and Manager subclasses we
# can just reuse that code by inheriting from the class Employee

# create subclass for Developer

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
		self.stock = int(self.stock * self.raise_amount)


class Developer(Employee):
	pass

# class <subclassName>(from what class we inherit):
# simply inheriting from the Employee class we inherited all of its
# functionality (attributes and methods) of Employee class

# create two new Employees
dev1 = Employee("Edward", "Teller", 4000)
dev2 = Employee("Corey", "Schafer", 5000)

print(dev1.email)
print(dev2.email)

# create two new Developers and pass in same information
dev1 = Developer("Edward", "Teller", 4000)
dev2 = Developer("Corey", "Schafer", 5000)

print(dev1.email)
print(dev2.email)

# now if we rerun this and print out the emails we can see that two developers
# created and access the attributes that where actually set in parent Employee
# class

# when we instanted our developers, it first looked in Developer class for the
# __init__ method where it not find since its empty so what python is doing then
# is walk up the chain of inheritance until it find what its looking for, now
# the chain is called the method resolution order

print(help(Developer))

# when we run help on Developer class we get all kinds of information here
# basically the method resolution order are the places where python searches for
# attributes and methods, so when we created two new developers it first looked
# in our Developer class for the __init__ method and when it didn't find it
# there then it went to the Employee class and it found it there and executed
# suppose if not found in the Employee class then the last place it looked is
# the builtins.object class since every class in python inherit from the
# builtins.object

print(dev1.stock)
dev1.applyRaise()
print(dev1.stock)

# let say that developer raise amount be 10%

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
		self.stock = int(self.stock * self.raise_amount)


class Developer(Employee):
	raise_amount = 1.10


dev1 = Developer("Edward", "Teller", 4000)
dev2 = Developer("Corey", "Schafer", 5000)

print(dev1.stock)
dev1.applyRaise()
print(dev1.stock)

# by changing raise_amount attribute in subclass Developer it
# don't have effect in Employee class raise_amount attribute
# we can make these changes to subclasses without worrying
# about breaking anything in parent class

# sometimes we want to initiate subclasses with more information than parent
# class handle, so let say when we create developer instance we also want to
# pass in their main programming language as an attribute but currently Employee
# class only accepts fname, lname and stock so if we also want to pass in
# programming language there then to get around this we have to give Developer
# class its own __init__ method

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
		self.stock = int(self.stock * self.raise_amount)


class Developer(Employee):
	raise_amount = 1.10

	# we might be tempted to copy code in __init__ method
	# of Employee class but to not repeat the code we let
	# the Employee __init__ method to handle fname, lname,
	# stock and let Developer __init__ to handle proglang
	def __init__(self, fname, lname, stock, proglang):
		super().__init__(fname, lname, stock)
		self.proglang = proglang


# super().__init__(fname, lname, stock) to Employee __init__ method and let
# Employee class handle those arguments

# there are multiple ways of doing this:
# super().__init__(fname, lname, stock)
# Employee.__init__(self, fname, lname, stock)
# both of these ways of calling the parents __init__ method will work
# it is recommanded to use super() because with single inheritance its
# maintainable but its neccessary once start using multiple inheritance

# when we instantiate developer it except proglang
dev1 = Developer("Edward", "Teller", 4000, "Python")
dev2 = Developer("Corey", "Schafer", 5000, "Kotlin")

print(dev1.email)
print(dev1.proglang)

# create subclass for Manager

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
		self.stock = int(self.stock * self.raise_amount)


class Developer(Employee):
	raise_amount = 1.10

	def __init__(self, fname, lname, stock, proglang):
		super().__init__(fname, lname, stock)
		self.proglang = proglang


class Manager(Employee):

	def __init__(self, fname, lname, stock, employees=None):
		super().__init__(fname, lname, stock)

		if employees is None:
			self.employees = []
		else:
			self.employees = employees


# pass the list of employees the Manager supervise as attribute
# set employees to an empty list if the argument is not provided
# and set them equal to that employees list if it is provided

# Note: not pass an empty list as default argument instead of None because
# never pass an mutable data type like a list or dict as default arguments

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
		self.stock = int(self.stock * self.raise_amount)


class Developer(Employee):
	raise_amount = 1.10

	def __init__(self, fname, lname, stock, proglang):
		super().__init__(fname, lname, stock)
		self.proglang = proglang


class Manager(Employee):

	def __init__(self, fname, lname, stock, employees=None):
		super().__init__(fname, lname, stock)

		if employees is None:
			self.employees = []
		else:
			self.employees = employees


	def addEmployee(self, employee):
		if employee not in self.employees:
			self.employees.append(employee)


	def remEmployee(self, employee):
		if employee in self.employees:
			self.employees.remove(employee)


	def printEmployees(self):
		for employee in self.employees:
			print(employee.employeeName())


# add and remove from the list of employees that the Manager supervise

# in Manager class, __init__ method accept fname, lname, stock and list of
# employees the manager supervises, the methods to add, remove and print out
# all the employees in that list

# create an instance of subclass Manager and the attributes and the methods
# inherit from the Employee class
man1 = Manager("Julian", "Schwinger", 9000, [dev1])

print(man1.email)

# print the employees the manager man1 supervise
man1.printEmployees()

# add an employee to the employees list of man1
man1.addEmployee(dev2)

# print the employees the manager man1 supervise
man1.printEmployees()

# remove an employee to the employees list of man1
man1.remEmployee(dev1)

# print the employees the manager man1 supervise
man1.printEmployees()

# just like with Developer class we can see how useful this actually is because
# all the code in Manager class is specific to what we want for a manager as the
# Developer class is specific to what we want for a developer and we are inherit
# all of the common code from the Employee class so we really get to reuse our
# code nicely here if we use subclassing correctly

# python has these two built-in functions called isinstance() and issubclass()
# isinstance() will tell us if an object is an instance of a class
print(isinstance(man1, Manager))	# True
print(isinstance(man1, Employee))	# True
print(isinstance(man1, Developer))	# False

# even though Developer and Manager both inherit from Employee class they are
# not part of each other inheritance

# issubclass() will tell us if a class is a subclass of another class
print(issubclass(Manager, Employee))	# True
print(issubclass(Developer, Manager))	# False
print(issubclass(Developer, Employee))	# True

# these built-in isinstance() and issubclass() functions may come and use in
# experementing with inheritance on your own

# Note: to learn more about subclassing look at Exception module of python
# whiskey library: class HTTPException(Exception)

"""
special methods:

* special methods are also called as magic methods

* special methods allows us to emulate build-in behavior within Python and its
  also how we implement operator-overloading

"""

# example of operator-overloading
print(1 + 2)			# 3
print('a' + 'b')		# ab

# when we add two integers together and two strings together the behaviour is
# different as the former result an integer whereas the later concatenate the
# two strings, so depending on what objects we working with the addition has
# different behaviour

# if we print the emp1 instance here
print(emp1)
# we get vague employee object and it will be nice if we could change this
# behaviour to print out something a bit user-friendly and thats what the
# special methods allow us to do, so by defining special methods able to
# change built-in behavior and operations

# special methods are always surrounded by double underscores (dunder), then
# dunder init means init surrounded by double underscores

# __init__ is special method we familiar with and its the first and most common
# special method when working with classes, as we learned __init__ method is
# implicitly called when we create our Employee objects and it sets all the
# attributes for us

# __repr__ is meant to be an unambigous representation of the object and should
# be used for debugging and logging (meant to be seen by other developers)

# __str__ is meant to be more readable representation of an object (meant to be
# used as display to an end user)

# let write the code for these and look at the difference, so first we want
# atleast have __repr__ method because with __repr__ and without __str__ then
# calling __str__ on an Employee would just use the __repr__ as fallback so its
# good to have this as minimum

# a good rule of thumb in creating this method is try to display something you
# can copy and paste back in the Python code that would recreate that same
# object

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
		self.stock = int(self.stock * self.raise_amount)

	# return a string that can be used to recreate the object
	def __repr__(self):
		return f'Employee("{self.fname}", "{self.lname}", {self.stock})'

	# display for the end user
	def __str__(self):
		return f'{self.employeeName} - {self.email}'


emp1 = Employee("Edward", "Teller", 4000)

# emp1 instance print __str__ by default
print(emp1)

# print __repr__ method
print(repr(emp1))

# print __str__ method
print(str(emp1))

# when we run repr(emp1) it actually calling emp1.__repr__() and similarly when
# we run str(emp1) it actually calling emp1.__str__()

# these two special methods __repr__ and __str__ allow us to change how our
# objects are printed and displayed

# unless we write complicated classes the methods __init__(), __repr__() and
# __str__() will be the ones we use most often

# special methods for arithmetic
# print(1 + 2) use a special method in the background __add__()
print(int.__add__(1, 2))     # access __add__() using int object
# strings are using there own __add__() method
print(str.__add__('a', 'b')) # access __add__() using str object
# we can customize how addition works for the objects by creating __add__()

# calculate total stocks by adding employees together

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
		self.stock = int(self.stock * self.raise_amount)

	# return a string that can be used to recreate the object
	def __repr__(self):
		return f'Employee("{self.fname}", "{self.lname}", {self.stock})'

	# display for the end user
	def __str__(self):
		return f'{self.employeeName} - {self.email}'

	# add stocks of employee instance
	def __add__(self, other):
		return self.stock + other.stock


emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

print(emp1 + emp2)

# read documentation for arithematic special methods
# examples:
print(len("institute"))
print("institute".__len__())

# compute total number of charactes in employee full name

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
		self.stock = int(self.stock * self.raise_amount)

	# return a string that can be used to recreate the object
	def __repr__(self):
		return f'Employee("{self.fname}", "{self.lname}", {self.stock})'

	# display for the end user
	def __str__(self):
		return f'{self.employeeName} - {self.email}'

	# add stocks of employee instance
	def __add__(self, other):
		return self.stock + other.stock

	# number of characters
	def __len__(self):
		return len(self.employeeName())


emp1 = Employee("Edward", "Teller", 4000)
emp2 = Employee("Corey", "Schafer", 5000)

print(len(emp1))
print(len(emp2))

# Note: refer datatime module for examples
