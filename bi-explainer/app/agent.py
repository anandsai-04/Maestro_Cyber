# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

import os
import google.auth

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


import pandas as pd
from pathlib import Path

def get_policy_features(policy_id: str) -> str:
    """Retrieves the engineered risk features for a specific cyber insurance policy.
    Use this tool to get the vendor control pressure, regulatory findings pressure, 
    and other key metrics for a policyholder to explain their BI loss tail risk.

    Args:
        policy_id: The ID of the policy (e.g., 'POL-001').

    Returns:
        A string containing the policy's risk features, or an error message if not found.
    """
    try:
        # Load the features dataset from the parent directory
        data_path = Path(__file__).parent.parent.parent / "data" / "09_cyber_pricing_features.csv"
        df = pd.read_csv(data_path)
        
        # In a real app we'd filter by policy_id. Our dataset uses random integer IDs or we might just have rows.
        # Let's just return a generic response or the first row if policy isn't explicitly matched.
        if "policy_id" in df.columns:
            policy_data = df[df["policy_id"] == policy_id]
            if not policy_data.empty:
                return policy_data.to_json(orient="records")
        
        # Fallback to returning a random representative sample of high risk
        high_risk = df.sort_values("vendor_control_pressure", ascending=False).head(1)
        return f"Policy {policy_id} not found. Here is a representative high-risk profile: " + high_risk.to_json(orient="records")
    except Exception as e:
        return f"Error retrieving data: {e}"


root_agent = Agent(
    name="bi_explainer",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert Cyber Actuary and Data Scientist. "
        "Your goal is to explain to non-technical stakeholders (like underwriters and brokers) "
        "WHY their Business Interruption (BI) loss premium is affected by specific risk factors, "
        "particularly Vendor Control Pressure and Regulatory Findings Pressure. "
        "Explain that the pricing engine uses a Monte Carlo simulation. "
        "Weak vendor controls lead to a higher probability of systemic ransomware events affecting multiple systems. "
        "Regulatory pressure indicates a higher likelihood of severe fines and prolonged forensics. "
        "Together, these drastically increase the 'TVaR 99%' tail risk, meaning the required capital reserve explodes, "
        "which multiplies the final Technical Premium. "
        "Use the get_policy_features tool to look up specific client data when asked."
    ),
    tools=[get_policy_features],
)

app = App(
    root_agent=root_agent,
    name="app",
)

