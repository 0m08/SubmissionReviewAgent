"""
Send notification emails via Gmail SMTP.
Supports threading via Message-ID, In-Reply-To, and References.
"""
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid, formatdate


def _get_smtp_credentials():
    """Return (user, password) or (None, None) if not configured."""
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_APP_PASSWORD", "").strip()
    if not user or not password:
        return None, None
    return user, password


def is_email_configured() -> bool:
    """Return True if SMTP_USER and SMTP_APP_PASSWORD are set."""
    user, password = _get_smtp_credentials()
    return user is not None and password is not None


def send_notification_email(
    to_email: str,
    subject: str,
    body_plain: str,
    reply_to_message_id: str | None = None,
    body_html: str | None = None,
) -> str | None:
    """
    Send an email via Gmail SMTP. Threading: pass reply_to_message_id for follow-ups.

    Args:
        to_email: Recipient address.
        subject: Subject line.
        body_plain: Plain-text body.
        reply_to_message_id: If set, sets In-Reply-To and References so the message
            appears in the same thread as the original. For the first message, omit.
        body_html: Optional HTML body (section headers/labels can be bold). If set,
            sent as multipart/alternative so HTML clients show formatted version.

    Returns:
        The Message-ID of the sent message (for use as reply_to_message_id in follow-ups),
        or None if sending failed or email is not configured.
    """
    user, password = _get_smtp_credentials()
    if not user or not password:
        return None

    msg_id = make_msgid(domain="coursegen")

    if body_html:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(body_plain, "plain", "utf-8"))
        msg.attach(MIMEText(body_html, "html", "utf-8"))
    else:
        msg = MIMEText(body_plain, "plain", "utf-8")

    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = msg_id

    if reply_to_message_id:
        msg["In-Reply-To"] = reply_to_message_id
        msg["References"] = reply_to_message_id

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_email], msg.as_string())
        return msg_id
    except Exception:
        return None
