# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample nested workflow demonstrating 3 layers of Workflows.

This example shows a multi-level workflow structure:
- Layer 0 (Root): Order Processing Pipeline
  - Layer 1: Validation Stage, Processing Stage, Notification Stage
    - Layer 2: Each stage contains sub-workflows
      - Layer 3: LlmAgent nodes
"""
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow import Edge
from google.adk.workflow import Workflow


check_inventory = LlmAgent(
    name="check_inventory",
    model="gemini-2.5-flash",
    instruction="""Check if the items in the order are in stock.
    Return 'IN_STOCK' if available, 'OUT_OF_STOCK' otherwise.""",
)

check_pricing = LlmAgent(
    name="check_pricing",
    model="gemini-2.5-flash",
    instruction="""Verify the pricing is correct for all items.
    Return 'PRICE_VALID' or 'PRICE_MISMATCH'.""",
)

validate_payment_method = LlmAgent(
    name="validate_payment_method",
    model="gemini-2.5-flash",
    instruction="""Validate the payment method provided.
    Return 'VALID' if acceptable, 'INVALID' otherwise.""",
)

verify_customer = LlmAgent(
    name="verify_customer",
    model="gemini-2.5-flash",
    instruction="""Verify customer account is in good standing.
    Return 'VERIFIED' or 'NEEDS_REVIEW'.""",
)

authorize_payment = LlmAgent(
    name="authorize_payment",
    model="gemini-2.5-flash",
    instruction="""Authorize the payment with the payment provider.
    Return 'AUTHORIZED' if successful, 'DECLINED' otherwise.""",
)

charge_customer = LlmAgent(
    name="charge_customer",
    model="gemini-2.5-flash",
    instruction="""Charge the customer's payment method.
    Return 'CHARGED' with transaction ID if successful.""",
)

pack_order = LlmAgent(
    name="pack_order",
    model="gemini-2.5-flash",
    instruction="""Pack the order items.
    Return 'PACKED' with package ID when complete.""",
)

generate_shipping_label = LlmAgent(
    name="generate_shipping_label",
    model="gemini-2.5-flash",
    instruction="""Generate a shipping label for the package.
    Return 'LABEL_GENERATED' with tracking number.""",
)

ship_order = LlmAgent(
    name="ship_order",
    model="gemini-2.5-flash",
    instruction="""Hand off the package to the shipping carrier.
    Return 'SHIPPED' with estimated delivery date.""",
)

send_confirmation_email = LlmAgent(
    name="send_confirmation_email",
    model="gemini-2.5-flash",
    instruction="""Send order confirmation email to customer.""",
)

send_sms_notification = LlmAgent(
    name="send_sms_notification",
    model="gemini-2.5-flash",
    instruction="""Send SMS notification to customer.""",
)


inventory_check_workflow = Workflow(
    name="inventory_check_workflow",
    edges=Edge.chain("START", check_inventory, check_pricing),
)

customer_check_workflow = Workflow(
    name="customer_check_workflow",
    edges=Edge.chain("START", verify_customer, validate_payment_method),
)

payment_processing_workflow = Workflow(
    name="payment_processing_workflow",
    edges=Edge.chain("START", authorize_payment, charge_customer),
)

shipping_workflow = Workflow(
    name="shipping_workflow",
    edges=Edge.chain("START", pack_order, generate_shipping_label, ship_order),
)

notification_delivery_workflow = Workflow(
    name="notification_delivery_workflow",
    edges=Edge.chain("START", send_confirmation_email, send_sms_notification),
)


validation_stage = Workflow(
    name="validation_stage",
    edges=Edge.chain("START", inventory_check_workflow, customer_check_workflow),
)

processing_stage = Workflow(
    name="processing_stage",
    edges=Edge.chain("START", payment_processing_workflow, shipping_workflow),
)

notification_stage = Workflow(
    name="notification_stage",
    edges=Edge.chain("START", notification_delivery_workflow),
)

root_agent = Workflow(
    name="order_processing_pipeline",
    edges=Edge.chain(
        "START",
        validation_stage,
        processing_stage,
        notification_stage,
    ),
)

app = App(
    name='nested_workflow',
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),
)
