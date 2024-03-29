import os
import configparser
import smtplib
from email.mime.text import MIMEText


def send_email(text, sub='send by program'):

    config_path = os.path.expanduser('~/.send_email.cfg')
    if not os.path.exists(config_path):
        print('setup configuration in ~/.send_email.cfg')

    config = configparser.ConfigParser()
    config.read(config_path)

    sender_add = config['sender']['add']
    sender_pwd = config['sender']['pwd']
    sender_smtp_add = config['sender']['smtp_add']
    sender_smtp_port = int(config['sender']['smtp_port'])
    # print(sender_add, sender_pwd, sender_smtp_add, sender_smtp_port)

    receiver_add = config['receiver']['add']
    # print(receiver_add)

    message = MIMEText(text, 'plain', 'utf-8')
    message['Subject'] = sub
    message['To'] = receiver_add
    message['From'] = sender_add

    smtp_server = smtplib.SMTP_SSL(sender_smtp_add, sender_smtp_port)
    smtp_server.login(sender_add, sender_pwd)

    smtp_server.sendmail(sender_add, [receiver_add], message.as_string())
    smtp_server.close()
    # print('Successfully the mail is sent')


if __name__ == '__main__':
    send_email('test')
