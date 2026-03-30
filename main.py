import threading
import bot
import gamma_bot

t1 = threading.Thread(target=bot.main)
t2 = threading.Thread(target=gamma_bot.main)

t1.start()
t2.start()

t1.join()
t2.join()
