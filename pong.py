
import pyautogui
import random as dom
import time as t
import turtle


window = turtle.Screen()
window.title("!!!PONG!!!!")
window.bgcolor('white')
window.setup(width=800, height=600)
window.tracer(0)
# turtle.bgpic("grass.gif")

score_one = 0
score_two = 0

list_of_x_easy = [0.2, 0.4, 0.6, 0.8, 1, -0.2, - 0.4, - 0.6, - 0.8, - 1]
list_of_y_easy = [0.2, 0.4, 0.6, 0.8, 1, -0.2, -0.4, - 0.6, - 0.8, - 1]
list_of_x_hard = [1.2, 1.4, 1.6, 1.8, 2, -1.2, -1.4, -1.6, - 1.8, - 2]
list_of_y_hard = [1.2, 1.4, 1.6, 1.8, 2, -1.2, -1.4, -1.6, -1.8, -2]

#  hard or easy mode

input_for_mode = pyautogui.confirm(
    "select a mode", buttons=['easy', 'hard', 'normal'], title="MODE SELECTION")


# info
info_game = turtle.Turtle()
info_game.speed(0)
info_game.color('red')
info_game.penup()
info_game.hideturtle()
info_game.goto(0, 0)
info_game.write("""
    THIS IS A SIMPLE GAME OF PONG/ BOUNCE OF THE BALL AND WIN
    SCORE LIMIT I SET TO 3 FOR NOW FOR FURTHER CHANGE GO TO 'BOUNCE' SECTION
    PLAYER ONE/LEFT OF THE SCREEN CONTROLS'w','s'
    PLAYER TWO CAN BE CONTROLLED BY 'r','f'.

             THIS TEXT WILL DISAPPEAR IN 10 SEC


""",
                align="center", font=("Courier", 10, "bold"))

# paddle player one // can also  use 'arrow','circle','turtle,'triangle'
paddle_one = turtle.Turtle()
paddle_one.speed(0)
paddle_one.shape('square')
paddle_one.color('black')
paddle_one.shapesize(stretch_wid=5, stretch_len=1)
paddle_one.penup()
paddle_one.goto(-350, 0)

# paddle player two // can also  use 'arrow','circle','turtle,'triangle'
paddle_two = turtle.Turtle()
paddle_two.speed(0)
paddle_two.shape('square')
paddle_two.color('black')
paddle_two.shapesize(stretch_wid=5, stretch_len=1)
paddle_two.penup()
paddle_two.goto(350, 0)


#  middle barrier
middle = turtle.Turtle()
middle.speed(0)
middle.shape('square')
middle.color('black')
middle.shapesize(stretch_wid=30, stretch_len=0.1)
middle.penup()
middle.goto(0, 0)

#  ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape('circle')
ball.color('red')
ball.dx = 0
ball.dy = 0
ball.penup()
ball.goto(0, 0)

if input_for_mode == "easy":
    ball.dx = dom.choice(list_of_x_easy)
    ball.dy = dom.choice(list_of_y_easy)
elif input_for_mode == "hard":
    ball.dx = dom.choice(list_of_x_hard)
    ball.dy = dom.choice(list_of_y_hard)
else:
    ball.dx = 0.8
    ball.dy = 0.8

#  pen turtle function

score_board = turtle.Turtle()
score_board.speed(0)
score_board.color('black')
score_board.penup()
score_board.hideturtle()
score_board.goto(0, 270)
score_board.write("PLAYER :> 0              PLAYER :> 0 ",
                  align="center", font=("Courier", 15, "bold"))


# all the functions


def paddle_one_move_up():
    y = paddle_one.ycor()
    y += 20
    paddle_one.sety(y)


def paddle_one_move_dn():
    y = paddle_one.ycor()
    y -= 20
    paddle_one.sety(y)


def paddle_two_move_up():
    y = paddle_two.ycor()
    y += 20
    paddle_two.sety(y)


def paddle_two_move_dn():
    y = paddle_two.ycor()
    y -= 20
    paddle_two.sety(y)


window.listen()
window.onkeypress(paddle_one_move_up, "w")
window.onkeypress(paddle_one_move_dn, "s")
window.onkeypress(paddle_two_move_up, "r")
window.onkeypress(paddle_two_move_dn, "f")

t.sleep(10)
info_game.clear()

# main loop
while True:
    window.update()

    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    if ball.ycor() > 290:
        ball.sety(290)
        ball.dy *= -1
    if ball.ycor() < -290:
        ball.sety(-290)
        ball.dy *= -1

    if ball.xcor() > 390:
        ball.goto(0, 0)
        ball.dx *= -1
        score_one += 1
        score_board.clear()
        score_board.write(f"PLAYER :{score_one}          PLAYER  :{score_two}",
                          align="center", font=("Courier", 15, "bold"))
    if ball.xcor() < -390:
        ball.goto(0, 0)
        ball.dx *= -1
        score_two += 1
        score_board.clear()
        score_board.write(f"PLAYER :{score_one}          PLAYER  :{score_two}",
                          align="center", font=("Courier", 15, "bold"))

#  bounce
    if (ball.xcor() > 340 and ball.xcor() < 350) and (ball.ycor() < paddle_two.ycor() + 40 and ball.ycor() > paddle_two.ycor() - 40):
        ball.setx(340)
        ball.dx *= -1
    if (ball.xcor() < -340 and ball.xcor() > -350) and (ball.ycor() < paddle_one.ycor() + 40 and ball.ycor() > paddle_one.ycor() - 40):
        ball.setx(-340)
        ball.dx *= -1

    if score_one == 3:
        score_board.clear()
        score_board.write(" PLAYER LEFT/ ONE WON ",
                          align="right", font=("Courier", 15, "bold"))
        t.sleep(5)
        break
    if score_two == 3:
        score_board.clear()
        score_board.write(" PLAYER RIGHT/TWO WON ",
                          align="left", font=("Courier", 15, "bold"))
        t.sleep(5)
        break
