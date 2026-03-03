#!/usr/bin/env python3
"""
BlackRoad Education — Stripe Integration
Handles course product creation, pricing, checkout sessions,
and webhook-driven enrollment for paid courses.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import stripe

# ─── Configuration ────────────────────────────────────────────────────────────

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# ─── Products & Prices ────────────────────────────────────────────────────────


def create_stripe_product(course_id: str, title: str,
                           description: Optional[str] = None,
                           image_url: Optional[str] = None) -> stripe.Product:
    """
    Create a Stripe Product for a BlackRoad course.

    ``course_id`` is stored as metadata so events can be correlated
    back to the LMS record without an extra database lookup.
    """
    kwargs: dict = {
        "name": title,
        "metadata": {"course_id": course_id},
    }
    if description:
        kwargs["description"] = description
    if image_url:
        kwargs["images"] = [image_url]
    return stripe.Product.create(**kwargs)


def create_stripe_price(product_id: str, amount_cents: int,
                         currency: str = "usd",
                         recurring: Optional[dict] = None) -> stripe.Price:
    """
    Create a Stripe Price for a product.

    Pass ``recurring={"interval": "month"}`` for subscription pricing.
    Omit for a one-time purchase.
    """
    kwargs: dict = {
        "product": product_id,
        "unit_amount": amount_cents,
        "currency": currency,
    }
    if recurring:
        kwargs["recurring"] = recurring
    return stripe.Price.create(**kwargs)


# ─── Checkout ─────────────────────────────────────────────────────────────────


def create_checkout_session(price_id: str, student_id: str,
                             course_id: str,
                             success_url: str,
                             cancel_url: str,
                             mode: str = "payment") -> stripe.checkout.Session:
    """
    Create a Stripe Checkout Session so a student can pay for a course.

    ``student_id`` and ``course_id`` are embedded as metadata so that
    the webhook handler can enroll the student on successful payment.

    *mode* is ``"payment"`` for one-time or ``"subscription"`` for recurring.
    """
    return stripe.checkout.Session.create(
        line_items=[{"price": price_id, "quantity": 1}],
        mode=mode,
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={"student_id": student_id, "course_id": course_id},
    )


# ─── Webhooks ─────────────────────────────────────────────────────────────────


def construct_webhook_event(payload: bytes, sig_header: str) -> stripe.Event:
    """
    Verify and construct a Stripe webhook Event from raw request bytes.

    Raises ``stripe.error.SignatureVerificationError`` if the signature
    is invalid.
    """
    return stripe.Webhook.construct_event(
        payload, sig_header, STRIPE_WEBHOOK_SECRET
    )


def handle_checkout_completed(event: stripe.Event) -> dict:
    """
    Process a ``checkout.session.completed`` event.

    Extracts ``student_id`` and ``course_id`` from the session metadata
    and returns them so the caller can trigger LMS enrollment.

    Returns::

        {
            "student_id": str,
            "course_id":  str,
            "session_id": str,
            "payment_status": str,
        }
    """
    session = event["data"]["object"]
    metadata = session.get("metadata", {})
    return {
        "student_id": metadata.get("student_id", ""),
        "course_id": metadata.get("course_id", ""),
        "session_id": session["id"],
        "payment_status": session.get("payment_status", ""),
    }


# ─── Convenience: sync a course to Stripe ─────────────────────────────────────


def sync_course_to_stripe(course_id: str, title: str, description: str,
                           amount_cents: int,
                           currency: str = "usd") -> dict:
    """
    One-call helper: create a Stripe Product **and** a one-time Price for
    a course, returning both IDs for storage in the LMS database.

    Returns::

        {
            "product_id": str,
            "price_id":   str,
            "amount_cents": int,
            "currency":   str,
        }
    """
    product = create_stripe_product(course_id, title, description)
    price = create_stripe_price(product["id"], amount_cents, currency)
    return {
        "product_id": product["id"],
        "price_id": price["id"],
        "amount_cents": amount_cents,
        "currency": currency,
    }
