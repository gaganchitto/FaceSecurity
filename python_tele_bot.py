from telegram import *
from telegram.ext import *
from time import  sleep

bot = Bot("1870501650:AAFkwc47g0XwFY5bzOy9v-EuNnCl21UNFEU")
# print(bot.getMe())
updater = Updater("1870501650:AAFkwc47g0XwFY5bzOy9v-EuNnCl21UNFEU",use_context = True)
dispatch = updater.dispatcher
shut = False



pic = 'D:/PYTHON_PROGRAMMING/IMPOSTER/imposter1.jpg'
chat_id = 988882140
# print(chat_id)
bot.send_photo(chat_id, photo=open(pic, 'rb'))
bot.send_message(chat_id, "Found Someone to Accessing your Laptop.\nYes for Allowing him\nNo for Shutting down the Laptop.")

def reply(update: Update, context: CallbackContext):
    bot.send_message(

        chat_id=update.effective_chat.id,
        text = "OK SIR I'M ALLOWING THAT PERSON TO ACCESS TO YOUR LAPTOP :-) "

    )

start_value = CommandHandler('Yes',reply)

dispatch.add_handler(start_value)

def reply1(update: Update, context: CallbackContext):
    global shut
    shut = True
    bot.send_message(

        chat_id=update.effective_chat.id,
        text = "OK SIR I'M SHUTTING DOWN YOUR LAPTOP :-) "

    )
start_value1 = CommandHandler('No',reply1)

dispatch.add_handler(start_value1)

updater.start_polling()
sleep(5)
updater.stop()