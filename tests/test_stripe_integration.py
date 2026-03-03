"""
Tests for BlackRoad Education — Stripe Integration.

All Stripe API calls are mocked via unittest.mock so tests run without
live credentials and without hitting the Stripe API.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

import stripe_integration as si


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _product(product_id="prod_test123", name="Python 101", course_id="crs-1"):
    p = MagicMock()
    p.__getitem__ = lambda s, k: {"id": product_id, "name": name,
                                  "metadata": {"course_id": course_id}}[k]
    p["id"] = product_id
    return p


def _price(price_id="price_test456", product_id="prod_test123",
           amount=4999, currency="usd"):
    p = MagicMock()
    p.__getitem__ = lambda s, k: {"id": price_id, "product": product_id,
                                  "unit_amount": amount, "currency": currency}[k]
    p["id"] = price_id
    return p


def _session(session_id="cs_test789", student_id="student-42",
             course_id="crs-1", payment_status="paid"):
    s = MagicMock()
    s.__getitem__ = lambda obj, k: {
        "id": session_id,
        "metadata": {"student_id": student_id, "course_id": course_id},
        "payment_status": payment_status,
    }[k]
    s.get = lambda k, default=None: {
        "metadata": {"student_id": student_id, "course_id": course_id},
        "payment_status": payment_status,
    }.get(k, default)
    return s


# ─── Product tests ────────────────────────────────────────────────────────────

def test_create_stripe_product():
    with patch("stripe.Product.create", return_value=_product()) as mock_create:
        product = si.create_stripe_product("crs-1", "Python 101", "Intro to Python")
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["name"] == "Python 101"
    assert call_kwargs["metadata"]["course_id"] == "crs-1"
    assert call_kwargs["description"] == "Intro to Python"


def test_create_stripe_product_no_description():
    with patch("stripe.Product.create", return_value=_product()) as mock_create:
        si.create_stripe_product("crs-2", "Go Basics")
    call_kwargs = mock_create.call_args.kwargs
    assert "description" not in call_kwargs


# ─── Price tests ─────────────────────────────────────────────────────────────

def test_create_stripe_price_one_time():
    with patch("stripe.Price.create", return_value=_price()) as mock_create:
        price = si.create_stripe_price("prod_test123", 4999)
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["unit_amount"] == 4999
    assert call_kwargs["currency"] == "usd"
    assert "recurring" not in call_kwargs


def test_create_stripe_price_recurring():
    with patch("stripe.Price.create", return_value=_price()) as mock_create:
        si.create_stripe_price("prod_test123", 999, recurring={"interval": "month"})
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["recurring"] == {"interval": "month"}


# ─── Checkout session tests ───────────────────────────────────────────────────

def test_create_checkout_session():
    with patch("stripe.checkout.Session.create",
               return_value=_session()) as mock_create:
        session = si.create_checkout_session(
            price_id="price_test456",
            student_id="student-42",
            course_id="crs-1",
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
        )
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["mode"] == "payment"
    assert call_kwargs["metadata"]["student_id"] == "student-42"
    assert call_kwargs["metadata"]["course_id"] == "crs-1"
    assert call_kwargs["success_url"] == "https://example.com/success"


# ─── Webhook tests ────────────────────────────────────────────────────────────

def test_construct_webhook_event():
    fake_event = MagicMock()
    with patch("stripe.Webhook.construct_event",
               return_value=fake_event) as mock_construct:
        event = si.construct_webhook_event(b"payload", "sig_header")
    mock_construct.assert_called_once_with(b"payload", "sig_header",
                                           si.STRIPE_WEBHOOK_SECRET)
    assert event is fake_event


def test_handle_checkout_completed():
    sess = _session(session_id="cs_test789", student_id="student-42",
                    course_id="crs-1", payment_status="paid")
    event = MagicMock()
    event.__getitem__ = lambda s, k: {"data": {"object": sess}}[k]

    result = si.handle_checkout_completed(event)
    assert result["student_id"] == "student-42"
    assert result["course_id"] == "crs-1"
    assert result["session_id"] == "cs_test789"
    assert result["payment_status"] == "paid"


# ─── sync_course_to_stripe tests ─────────────────────────────────────────────

def test_sync_course_to_stripe():
    prod = _product(product_id="prod_abc", course_id="crs-3")
    price_ = _price(price_id="price_xyz", product_id="prod_abc", amount=9900)

    with patch("stripe.Product.create", return_value=prod), \
         patch("stripe.Price.create", return_value=price_):
        result = si.sync_course_to_stripe(
            course_id="crs-3",
            title="Advanced Python",
            description="Deep dive",
            amount_cents=9900,
        )

    assert result["product_id"] == "prod_abc"
    assert result["price_id"] == "price_xyz"
    assert result["amount_cents"] == 9900
    assert result["currency"] == "usd"
