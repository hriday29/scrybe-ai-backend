# notifier.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from logger_config import log
import config
from datetime import datetime

def send_daily_briefing(new_signals: list, closed_trades: list):
    """Constructs and sends the daily email summary."""
    
    if not new_signals and not closed_trades:
        log.info("No new signals or closed trades to report. Skipping email.")
        return

    sender_email = config.GMAIL_ADDRESS
    password = config.GMAIL_APP_PASSWORD
    receiver_emails = config.BETA_TESTER_EMAILS

    if not all([sender_email, password, receiver_emails]):
        log.error("Email credentials or recipient list not configured. Cannot send email.")
        return

    # Create the email content
    subject = f"Scrybe AI Daily Briefing: {datetime.now().strftime('%d %b, %Y')}"
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"Scrybe AI <{sender_email}>"
    message["To"] = ", ".join(receiver_emails)

    # --- Build the HTML Body ---
    html_body = """
    <html>
      <head>
        <style>
          body { font-family: sans-serif; background-color: #0A0F1E; color: #E2E8F0; padding: 20px; }
          .container { max-width: 600px; margin: auto; background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 20px; }
          h2 { color: #FFFFFF; border-bottom: 1px solid #475569; padding-bottom: 10px; }
          .trade-item { margin-bottom: 15px; padding: 10px; border-radius: 6px; }
          .buy { background-color: #10B981; color: #FFFFFF; }
          .sell { background-color: #EF4444; color: #FFFFFF; }
          .closed { background-color: #475569; color: #E2E8F0; }
          p { margin: 5px 0; }
          .footer { margin-top: 20px; text-align: center; font-size: 12px; color: #94A3B8; }
        </style>
      </head>
      <body>
        <div class="container">
    """
    
    if new_signals:
        html_body += "<h2>New Trade Signals</h2>"
        for signal in new_signals:
            signal_class = "buy" if signal['signal'] == 'BUY' else 'sell'
            score = signal.get('scrybeScore', 0)
            score_text = f"+{score}" if score > 0 else str(score)
            
            html_body += f"""
            <div class="trade-item {signal_class}">
                <p><strong>{signal['signal']} Signal: {signal['ticker']}</strong></p>
                <p><strong>Scrybe Score: {score_text}</strong></p>
                <p>Confidence: {signal['confidence']}</p>
            </div>
            """

    if closed_trades:
        html_body += "<h2>Closed Trades Today</h2>"
        for trade in closed_trades:
             html_body += f"""
            <div class="trade-item closed">
              <p><strong>{trade['ticker']} ({trade['signal']})</strong></p>
              <p>Outcome: {trade['closing_reason']}</p>
              <p>Return: {trade['return_pct']:.2f}%</p>
            </div>
            """

    html_body += """
          <div class="footer">
            <p>This is an automated report from Scrybe AI. All signals are for informational purposes only.</p>
          </div>
        </div>
      </body>
    </html>
    """
    
    message.attach(MIMEText(html_body, "html"))

    # Send the email
    try:
        log.info(f"Connecting to SMTP server to send daily briefing...")
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_emails, message.as_string())
        server.close()
        log.info(f"✅ Daily briefing email sent successfully to {len(receiver_emails)} user(s).")
    except Exception as e:
        log.error(f"Failed to send email: {e}", exc_info=True)