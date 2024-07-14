-- File: guess.adb
-- Name: D.Saravanan
-- Date: 10/07/2024
-- Program for number guessing game

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics.Discrete_Random;

procedure Guess is
    type randRange is range 1..100;
    package Rand_Int is new ada.numerics.discrete_random(randRange);
    use Rand_Int;
    gen : Generator;
    num : randRange;
    incorrect: Boolean := True;
    guess: randRange;

begin
    reset(gen);
    num := random(gen);
    while incorrect loop
        Put_Line("Guess a number between 1 and 100");
        declare
            guess_str : String := Get_Line(Current_Input);
        begin
            guess := randRange'Value(guess_str);
        end;
        if guess < num then
            Put_line("Too low");
        elsif guess > num then
            Put_line("Too high");
        else
            incorrect := False;
        end if;
    end loop;
    Put_line("That's right");

end Guess;
