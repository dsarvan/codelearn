-- File: check_positive.adb
-- Name: D.Saravanan
-- Date: 11/07/2024
-- Program check an integer value is positive or not

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Integer_Text_IO; use Ada.Integer_Text_IO;

procedure Check_Positive is
    N : Integer;
begin
    -- Put a String
    Put("Enter an integer value: ");

    -- Reads in an integer value
    Get(N);

    -- Put an Integer
    Put(N);

    if N > 0 then
        Put_Line(" is a positive number");
    else
        Put_Line(" is not a positive number");
    end if;
end Check_Positive;
