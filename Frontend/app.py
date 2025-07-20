

from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk
import google.generativeai as genai
from flask import Flask, request, redirect, url_for, render_template, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import ast

# Download NLTK data with error handling
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"NLTK download error: {e}")

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBnkaJOe-xX_5YVOD-nn-qH5KELvbFYueg"
genai.configure(api_key=GEMINI_API_KEY)

# Set up the model - use a valid model name
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize intent_responses with a default empty dictionary
intent_responses = {}

# Load ML model and related files with error handling
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    
    # Load the expanded dataset with better error handling
    try:
        with open('intent_ds.pkl', 'rb') as f:
            intent_responses = pickle.load(f)
        print("Successfully loaded expanded_customer_support_dataset.pkl")
    except Exception as e:
        print(f"Error loading expanded dataset: {e}")
        # Create a basic fallback dataset in case loading fails
        intent_responses = {
            "Return_Request": "You can return the product within 30 days of delivery.",
            "Technical_Support": "Please reset the device using the settings menu.",
            "fallback": "I'm not sure I understand your question. Can you please provide more details?"
        }
    
    print("Successfully loaded all SVM model files")
    models_loaded = True
except Exception as e:
    print(f"Error loading model files: {e}")
    tfidf = None
    le = None
    svm = None
    intent_responses = {}
    models_loaded = False

# Define hard-coded intent responses as fallback in case dataset loading fails
# This ensures we always have some responses available
INTENT_RESPONSES = {
    "Return_Request": "You can return the product within 30 days of delivery.",
    "Technical_Support": "Please reset the device using the settings menu.",
    "Change_Address": "After shipping, the address can't be changed. Contact support for help.",
    "Order_Status": "Your package is on its way and should arrive soon.",
    "Shipping_Problem": "We're sorry for the delay. Your item will be delivered shortly.",
    "Account_Help": "Please use the 'Forgot Password' option to reset it.",
    "Refund_Status": "Your refund will reflect in your account within 3-5 days.",
    "Payment_Issue": "If money was deducted, it will be refunded within 7 days.",
    "Prime_Membership": "You can cancel your Prime from 'Manage Prime Membership'.",
    "Product_Inquiry": "Yes, this product is waterproof and includes a 1-year warranty.",
    "Cancel_Order": "You can cancel within 30 minutes of placing the order if it hasn't shipped.",
    "Exchange_Request": "To exchange an item, go to 'Your Orders' and select 'Return or Replace Items'.",
    "Product_Not_Received": "Please check with neighbors or contact support if not received.",
    "Gift_Card_Help": "Redeem gift cards from the 'Gift Cards' section in your account.",
    "Technical_Connectivity": "Please try disconnecting and reconnecting your device to the network, then restart it to resolve connectivity issues.",
    "Technical_Battery": "For battery issues, try calibrating your battery by fully draining it and then charging to 100% without interruption.",
    "Technical_Software_Update": "Please update to the latest software version by going to Settings > System > Software Update for optimal performance.",
    "Technical_App_Issues": "If the app is crashing, try clearing the cache in Settings > Apps > [App Name] > Storage > Clear Cache.",
    "Technical_Screen_Problems": "For screen issues, try adjusting brightness settings or enabling auto-brightness in the display settings menu.",
    "Technical_Sound_Issues": "If you're experiencing audio problems, check volume settings and ensure the device isn't in silent or Do Not Disturb mode.",
    "Technical_Overheating": "To address overheating, close background apps, remove any case temporarily, and avoid charging while using processor-intensive applications.",
    "Technical_Slow_Performance": "If your device is running slowly, try closing background apps and clearing temporary files from the storage settings.",
    "Return_Damaged_Item": "For damaged items, please submit photos of the damage through the Returns Center for an expedited return process.",
    "Return_Wrong_Size": "For size-related returns, we offer free return shipping. You can print a return label from your order details page.",
    "Return_Missing_Parts": "If your product has missing parts, contact us immediately and we can ship the missing components or process a full return.",
    "Return_Changed_Mind": "If you've changed your mind, unopened items can be returned within 45 days for a full refund to your original payment method.",
    "Return_Defective_Product": "For defective products, we offer immediate replacement or full refund including any shipping costs incurred.",
    "Return_Gift_Return": "Gift returns are processed as store credit at the current selling price without notifying the gift giver.",
    "Return_Outside_Window": "For returns outside our standard window, please contact customer service as exceptions may be possible for certain circumstances.",
    "Return_International": "International returns require a different process. Please request an international return authorization before shipping back items.",
    "Delivery_Options": "We offer standard (3-5 days), expedited (2 days), and premium (next-day) delivery options at checkout.",
    "Delivery_Tracking_Issues": "If your tracking hasn't updated in 24 hours, please contact us for a detailed investigation with the carrier.",
    "Delivery_Partial_Order": "For partial deliveries, the remaining items will ship separately once they become available, at no extra shipping cost.",
    "Delivery_Weekend": "Weekend deliveries are available in select areas for orders placed before Friday at 12 PM. Check eligibility during checkout.",
    "Delivery_Rural_Areas": "For rural locations, please allow 1-2 additional days beyond the estimated delivery date shown at checkout.",
    "Delivery_Signature_Required": "Your package requires a signature upon delivery. If you won't be available, you can pre-authorize release through the carrier's website.",
    "Delivery_Rescheduling": "To reschedule a delivery, use the tracking link in your shipping confirmation email to select a new delivery date.",
    "Delivery_Instructions": "You can add delivery instructions for the driver in your account settings under 'Address Book' for all future deliveries."
}

# Enhanced intent matching function with expanded keywords
def match_intent(user_input):
    """Match user input to predefined intents using comprehensive keyword matching"""
    user_input = user_input.lower()
    
    # Expanded intent keywords with more variations and common phrasings
    intent_keywords = {
        "Return_Request": [
            "return", "give back", "refund", "send back", "take back", "don't want anymore", 
            "return policy", "return procedure", "how to return", "want my money back", 
            "return the item", "return the product", "return the order", "don't like it", 
            "wrong item", "damaged item", "defective product", "not as described", 
            "changed my mind", "return address", "return label", "return window", 
            "return timeframe", "refund policy", "refund process", "refund timeframe", 
            "return instructions", "send it back", "return eligibility", "return shipping"
        ],
        
        "Technical_Support": [
            "technical", "support", "help", "not working", "broken", "fix", "issue", "problem",
            "troubleshoot", "malfunctioning", "error", "error message", "glitch", "bug", 
            "tech support", "technical assistance", "technical help", "device problem", 
            "product help", "setup help", "installation issue", "configuration problem", 
            "troubleshooting", "system error", "doesn't turn on", "won't connect", 
            "keeps crashing", "freezing", "screen black", "battery issue", "connection problem", 
            "software update", "hardware problem", "repair options", "diagnostic", 
            "factory reset", "user guide", "manual", "instructions", "setup wizard",
            "device not responding", "tech issue", "how to fix", "software issue", "firmware"
        ],
        
        "Change_Address": [
            "address", "change address", "shipping address", "delivery address", "wrong address",
            "update address", "edit address", "modify address", "incorrect address", 
            "new address", "address change", "delivery location", "ship to", "send to", 
            "shipping destination", "delivery destination", "moved", "moved house", 
            "moved apartment", "moved office", "relocating", "relocation", "new home", 
            "new apartment", "new house", "change delivery", "update delivery", 
            "address correction", "fix address", "different address", "alternative address", 
            "address typo", "change shipping", "address mistake", "address error"
        ],
        
        "Order_Status": [
            "order status", "where is", "tracking", "shipped", "delivered", "package", "arrival",
            "delivery status", "shipment status", "order progress", "track package", 
            "track order", "track delivery", "delivery date", "estimated delivery", 
            "delivery time", "arrival time", "shipping status", "order information", 
            "when will arrive", "when will it come", "when will it get here", "arrival date", 
            "is it shipped", "has it shipped", "dispatch status", "order confirmation", 
            "order tracking", "shipping notification", "delivery notification", "delivery update", 
            "delivery progress", "when should I expect", "package location", "courier tracking", 
            "delivery estimate", "order shipped", "order processed", "processing status", 
            "order number", "order lookup", "check my order"
        ],
        
        "Shipping_Problem": [
            "shipping", "delay", "late", "delivery issue", "not arrived", "shipping delay", 
            "delayed package", "late delivery", "late package", "missed delivery", 
            "wrong delivery", "delivery exception", "delivery problem", "shipping error", 
            "package stuck", "shipping stuck", "package lost", "lost in transit", "transit issue", 
            "delivery postponed", "postponed shipping", "missed the delivery", "reschedule delivery", 
            "delivery attempt failed", "failed delivery", "delivery rescheduling", "slow shipping", 
            "carrier issue", "courier problem", "shipping carrier", "delivery service", 
            "package damaged", "damaged in transit", "delivery date changed", "delivery time changed", 
            "incomplete delivery", "package returned to sender", "wrong shipping method"
        ],
        
        "Account_Help": [
            "account", "login", "password", "forgot", "can't access", "sign in", "sign up", 
            "registration", "create account", "account creation", "account settings", 
            "account details", "profile settings", "profile details", "account management", 
            "manage account", "account information", "sign in problem", "login problem", 
            "login issue", "can't login", "unable to login", "reset password", "change password", 
            "password reset", "recover account", "account recovery", "account verification", 
            "verify account", "account security", "authentication", "two-factor", "2FA", 
            "security question", "username", "email change", "change email", "account locked", 
            "locked out", "account suspension", "suspended account", "disabled account"
        ],
        
        "Refund_Status": [
            "refund status", "money back", "reimbursement", "credit", "refund progress", 
            "refund tracking", "track refund", "check refund", "refund information", 
            "refund processed", "refund completed", "refund initiated", "refund issued", 
            "refund pending", "awaiting refund", "expecting refund", "refund delay", 
            "delayed refund", "refund timeframe", "when refund", "refund time", "refund period", 
            "refund method", "how long refund", "refund duration", "where is my refund", 
            "refund confirmation", "confirm refund", "refund receipt", "refund to card", 
            "refund to account", "original payment method", "refund to wallet", "partial refund", 
            "refund amount", "full refund", "refund approved", "refund denied", "refund rejected"
        ],
        
        "Payment_Issue": [
            "payment", "charged", "card", "debit", "transaction", "money", "payment failed", 
            "payment declined", "payment error", "payment problem", "payment issue", 
            "double charged", "charged twice", "incorrect charge", "wrong charge", 
            "extra charge", "unauthorized charge", "unexpected charge", "charge dispute", 
            "dispute transaction", "payment method", "card declined", "card rejected", 
            "insufficient funds", "bank declined", "payment authorization", "payment verification", 
            "verify payment", "payment confirmation", "billing issue", "billing problem", 
            "payment processing", "payment gateway", "payment system", "payment portal", 
            "payment page", "checkout issue", "checkout problem", "payment failure"
        ],
        
        "Prime_Membership": [
            "prime", "membership", "subscription", "monthly fee", "cancel prime", 
            "prime benefits", "prime advantages", "prime features", "prime cost", "prime price", 
            "prime renewal", "renew prime", "prime expiration", "prime expired", 
            "prime membership fee", "prime subscription fee", "cancel subscription", 
            "end subscription", "terminate subscription", "stop subscription", "unsubscribe", 
            "prime trial", "free trial", "trial period", "prime discount", "prime shipping", 
            "prime delivery", "prime video", "prime music", "prime reading", "student prime", 
            "family plan", "household members", "shared benefits", "membership sharing", 
            "prime day", "annual membership", "monthly membership", "upgrade membership", 
            "downgrade membership", "membership status"
        ],
        
        "Product_Inquiry": [
            "product", "details", "specs", "waterproof", "warranty", "features", 
            "product information", "product details", "product description", "product specifications", 
            "product features", "product dimensions", "product size", "product color", 
            "product version", "product model", "compatible with", "compatibility", 
            "product comparison", "compare products", "product difference", "similar products", 
            "alternatives", "newer model", "older model", "latest version", "product materials", 
            "product components", "product contents", "product included", "product quality", 
            "product durability", "product lifespan", "product reliability", "product reviews", 
            "customer reviews", "product ratings", "specifications sheet", "user manual", 
            "instruction manual", "product guide", "how to use", "usage instructions"
        ],
        
        "Cancel_Order": [
            "cancel", "stop order", "don't want", "cancel order", "order cancellation", 
            "cancel purchase", "stop purchase", "revoke order", "withdraw order", "undo order", 
            "change mind", "changed mind", "no longer want", "don't need anymore", 
            "cancel before shipping", "cancel before delivery", "prevent shipment", 
            "prevent delivery", "cancel confirmation", "cancellation policy", 
            "cancellation window", "cancellation period", "cancellation fee", 
            "cancellation charge", "order modification", "modify order", "remove item", 
            "delete order", "void order", "void transaction", "cancellation request", 
            "request cancellation", "order deletion", "unwanted order", "mistaken order", 
            "accidental order", "wrong order"
        ],
        
        "Exchange_Request": [
            "exchange", "replace", "replacement", "different size", "swap", "switch item", 
            "exchange process", "exchange policy", "exchange window", "exchange period", 
            "item exchange", "product exchange", "size exchange", "color exchange", 
            "model exchange", "exchange for different", "wrong size", "incorrect size", 
            "doesn't fit", "too small", "too big", "wrong color", "different color", 
            "wrong model", "different model", "exchange procedure", "how to exchange", 
            "exchange instructions", "exchange eligibility", "eligible for exchange", 
            "exchange timeframe", "exchange for same item", "direct replacement", 
            "exchange options", "exchange alternatives", "like-for-like exchange", 
            "straight exchange", "exchange process", "exchange for store credit"
        ],
        
        "Product_Not_Received": [
            "not received", "didn't get", "missing package", "never arrived", "no delivery", 
            "package missing", "delivery missing", "not delivered", "failed to deliver", 
            "didn't arrive", "hasn't arrived", "never delivered", "delivery failure", 
            "missing shipment", "missing order", "lost order", "lost package", 
            "lost delivery", "undelivered", "delivery not made", "no sign of package", 
            "package whereabouts", "where is my package", "where is my order", 
            "package not found", "unfulfilled order", "incomplete delivery", 
            "delivery not completed", "expected delivery missed", "absent delivery", 
            "delivery not attempted", "tracking says delivered", "delivered but not received", 
            "fake delivery", "delivery proof", "delivery confirmation", "signed by someone else", 
            "wrong delivery location", "delivered to neighbor", "delivered to wrong address",
            "stolen package", "package theft"
        ],
        
        "Gift_Card_Help": [
            "gift card", "redeem", "code", "voucher", "gift certificate", "gift voucher", 
            "gift code", "promotional code", "promo code", "coupon", "discount code", 
            "gift card balance", "check balance", "remaining balance", "gift card value", 
            "add gift card", "apply gift card", "use gift card", "activate gift card", 
            "gift card activation", "register gift card", "gift card registration", 
            "gift card expired", "gift card expiration", "gift card validity", 
            "invalid gift card", "gift card not working", "gift card problem", 
            "gift card error", "gift card rejected", "combine gift cards", 
            "multiple gift cards", "partial gift card", "gift card purchase", 
            "buy gift card", "send gift card", "e-gift card", "digital gift card", 
            "physical gift card", "lost gift card", "stolen gift card", "gift card replacement"
        ],
        "Technical_Connectivity": [
            "wifi issue", "internet connection", "bluetooth problem", "won't connect", "connection drops",
            "network issue", "wifi drops", "no internet", "pairing problem", "wireless issue",
            "connection error", "offline", "signal strength", "weak signal", "disconnecting"
        ],
        
        "Technical_Battery": [
            "battery drain", "battery life", "not charging", "dies quickly", "won't hold charge",
            "battery percentage", "rapid drain", "charging problem", "battery swelling", "power issue",
            "short battery life", "battery calibration", "slow charging", "overheating while charging"
        ],
        
        # Return subcategories
        "Return_Damaged_Item": [
            "arrived broken", "damaged in shipping", "packaging damaged", "arrived cracked", 
            "dented product", "damaged during transit", "scratched", "chipped", "torn", "smashed"
        ],
        
        "Return_Missing_Parts": [
            "incomplete package", "missing components", "not all parts", "missing accessories",
            "missing manual", "missing pieces", "part missing", "incomplete set", "missing hardware"
        ],
        
        # Delivery subcategories
        "Delivery_Options": [
            "delivery choices", "shipping options", "faster shipping", "expedited options", 
            "next day delivery", "same day shipping", "shipping speed", "quick delivery", 
            "standard delivery", "express delivery", "overnight shipping", "delivery timeframes"
        ],
        
        "Delivery_Rescheduling": [
            "change delivery date", "reschedule shipment", "new delivery time", "delivery day change",
            "different delivery day", "postpone delivery", "expedite delivery", "delivery adjustment",
            "can't be home", "delivery window", "delivery appointment", "alternate delivery date"
        ]
    }
    
    # Check if the user input contains keywords for any intent
    matched_intents = {}
    for intent, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword in user_input:
                if intent in matched_intents:
                    matched_intents[intent] += 1
                else:
                    matched_intents[intent] = 1
    
    # Return the intent with the most keyword matches
    if matched_intents:
        return max(matched_intents, key=matched_intents.get)
    return None
    

# Preprocessing function with error handling
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Use safer tokenization approach to avoid punkt_tab issue
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # If punkt_tab fails, fall back to basic splitting
            tokens = text.split()
            
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except LookupError:
            # If stopwords fails, proceed without stopword removal
            pass
            
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        # Return original text if processing fails
        return text


def format_gemini_response(response_text):
    """Format Gemini response to be more structured with proper line breaks for readability"""
    # Remove source attribution if present
    response_text = re.sub(r'\[Source: .*?\]$', '', response_text)
    
    # Replace numbered list items with proper formatting
    # Add line breaks before numbered items for better structure
    formatted_text = re.sub(r'(\d+\.\s*\*\*[^*]+\*\*)', r'\n\1', response_text)
    
    # Replace ** with line breaks to create visual separation for important points
    formatted_text = formatted_text.replace('**', '\n')
    
    # Clean up excessive line breaks
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
    
    # Ensure paragraphs have proper spacing
    formatted_text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', formatted_text)
    
    # Structure lists better
    formatted_text = re.sub(r'\n(\d+\.)', r'\n\n\1', formatted_text)
    
    return formatted_text.strip()


def get_gemini_response(query, conversation_history=None):
    try:
        # Enhanced system prompt for better mobile support responses
        system_prompt = """You are a helpful customer service assistant specializing in mobile devices and electronics. 
        Be polite, concise, and helpful. Provide clear and accurate information.
        
        Format your responses with these guidelines:
        1. Start with a brief acknowledgment of the customer's issue
        2. Keep paragraphs to 4-5 sentences maximum
        3. Present troubleshooting steps in a clear, structured manner
        4. Be direct and solution-oriented
        
        For sound-related issues with mobile phones, consider common problems like:
        - Speaker/earpiece problems (cleaning, volume settings, audio balance)
        - Notification sound issues (settings, silent mode, do not disturb)
        - Call sound problems (network issues, microphone blockage)
        
        If you don't know something, admit it and offer to connect the user with a human agent.
        """
        
        # Prepare generation config - slightly lowered temperature for more factual responses
        generation_config = {
            "temperature": 0.5,  # Reduced for more consistent answers
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        if conversation_history:
            formatted_history = []
            for msg in conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                formatted_history.append({"role": role, "parts": [msg["parts"][0]]})
            
            chat = model.start_chat(history=formatted_history)
            response = chat.send_message(query, generation_config=generation_config)
        else:
            response = model.generate_content(
                f"{system_prompt}\n\nUser query: {query}",
                generation_config=generation_config
            )

        if hasattr(response, 'text'):
            return format_gemini_response(response.text)
        elif hasattr(response, 'parts') and response.parts:
            return format_gemini_response(response.parts[0].text)
        else:
            return format_gemini_response(str(response))

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "I'm having trouble connecting to my knowledge base right now. Please try asking a different question or contact customer support."


# Database setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS conversations 
                 (id INTEGER PRIMARY KEY, username TEXT, message TEXT, sender TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_info 
                 (username TEXT PRIMARY KEY, name TEXT, product TEXT, order_number TEXT, 
                 issue_type TEXT, contact_method TEXT)''')
    conn.commit()
    conn.close()

# Save conversation to the database
def save_conversation(username, message, sender):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO conversations (username, message, sender) VALUES (?, ?, ?)", (username, message, sender))
    conn.commit()
    conn.close()

# Save user information to the database
def save_user_info(username, name, product, order_number, issue_type, contact_method):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO user_info 
                 (username, name, product, order_number, issue_type, contact_method) 
                 VALUES (?, ?, ?, ?, ?, ?)''', 
              (username, name, product, order_number, issue_type, contact_method))
    conn.commit()
    conn.close()

# Retrieve all conversations for a user
def get_conversations(username):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT sender, message FROM conversations WHERE username=?", (username,))
    conversations = c.fetchall()
    conn.close()
    return conversations

# Retrieve conversation history for Gemini context
def get_conversation_history(username, limit=10):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""
        SELECT sender, message FROM conversations 
        WHERE username=? 
        ORDER BY id DESC LIMIT ?
    """, (username, limit))
    recent_conversations = c.fetchall()
    conn.close()
    
    # Format for Gemini API (reverse to get chronological order)
    history = []
    for sender, message in reversed(recent_conversations):
        role = "user" if sender == "user" else "model"
        history.append({"role": role, "parts": [message]})
    
    return history if history else None

# Get user information
def get_user_info(username):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT name, product, order_number, issue_type, contact_method FROM user_info WHERE username=?", (username,))
    user_info = c.fetchone()
    conn.close()
    return user_info

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Validate input
        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('signup.html')

        # Check if password is strong enough
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('signup.html')

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Try to insert into the database
        try:
            with sqlite3.connect('database.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                session['username'] = username
                flash('Signup successful! Please complete setup.', 'success')
                return redirect(url_for('chatbot'))
        except sqlite3.IntegrityError:
            flash('Username already exists! Please choose a different one.', 'error')
            return render_template('signup.html')

    return render_template('signup.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Validate input
        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('login.html')

        # Fetch user info from database
        with sqlite3.connect('database.db') as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username=?", (username,))
            result = c.fetchone()

        # Check password
        if result and check_password_hash(result[0], password):
            session['username'] = username
            user_info = get_user_info(username)

            if user_info:
                session.update({
                    'name': user_info[0],
                    'product': user_info[1],
                    'order_number': user_info[2],
                    'issue_type': user_info[3],
                    'contact_method': user_info[4]
                })
                return redirect(url_for('chatbot'))
            else:
                # First time user, redirect to setup page
                return redirect(url_for('chatbot'))
        else:
            flash('Invalid username or password!', 'error')

    return render_template('login.html')


# Setup route for new users
@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        product = request.form.get('product', '').strip()
        order_number = request.form.get('order_number', '').strip()
        issue_type = request.form.get('issue_type', '').strip()
        contact_method = request.form.get('contact_method', '').strip()
        
        # Save user info to database
        save_user_info(session['username'], name, product, order_number, issue_type, contact_method)
        
        # Update session with user info
        session.update({
            'name': name,
            'product': product,
            'order_number': order_number,
            'issue_type': issue_type,
            'contact_method': contact_method
        })
        
        flash('Setup complete!', 'success')
        return redirect(url_for('chatbot'))
    
    return render_template('setup.html')

# Chatbot route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get user info if not in session
    if 'name' not in session:
        user_info = get_user_info(session['username'])
        if user_info:
            session.update({
                'name': user_info[0],
                'product': user_info[1],
                'order_number': user_info[2],
                'issue_type': user_info[3],
                'contact_method': user_info[4]
            })
        else:
            return redirect(url_for('setup'))
    
    # Use ML model to classify intent
    intent = None
    response = None
    
    if request.method == 'POST':
        user_message = request.form.get('message', '')
        
        # Save the user message
        save_conversation(session['username'], user_message, 'user')
        
        # Get conversation history for Gemini context
        conversation_history = get_conversation_history(session['username'])
        
        # Process message and get response
        if models_loaded:
            try:
                # Use the ML model to predict intent
                processed_input = preprocess_text(user_message)
                vectorized_input = tfidf.transform([processed_input])
                intent_id = svm.predict(vectorized_input)[0]
                intent = le.inverse_transform([intent_id])[0]
                
                # Get the response from the learned dataset
                if intent in intent_responses:
                    response = intent_responses[intent]
                else:
                    # Fallback to keyword matching
                    keyword_intent = match_intent(user_message)
                    if keyword_intent and keyword_intent in INTENT_RESPONSES:
                        intent = keyword_intent
                        response = INTENT_RESPONSES[keyword_intent]
                    else:
                        # Use Gemini API as a final fallback
                        response = get_gemini_response(user_message, conversation_history)
            except Exception as e:
                print(f"ML prediction error: {e}")
                intent = match_intent(user_message)
                if intent and intent in INTENT_RESPONSES:
                    response = INTENT_RESPONSES[intent]
                else:
                    # Use Gemini API as a final fallback
                    response = get_gemini_response(user_message, conversation_history)
        else:
            # If models not loaded, use keyword matching
            intent = match_intent(user_message)
            if intent and intent in INTENT_RESPONSES:
                response = INTENT_RESPONSES[intent]
            else:
                # Use Gemini API as a final fallback
                response = get_gemini_response(user_message, conversation_history)
        
        # Personalize response with user info if available
        if response:
            # Customize response with user information
            response = response.replace("[NAME]", session.get('name', 'valued customer'))
            response = response.replace("[PRODUCT]", session.get('product', 'your product'))
            response = response.replace("[ORDER]", session.get('order_number', 'your order'))
            
            # Enhance response with more personalization
            if "Return" in intent and session.get('product'):
                response += f" For your {session.get('product')}, please ensure it's in the original packaging if possible."
            
            if "Technical" in intent and session.get('product'):
                response += f" These steps are specifically helpful for {session.get('product')} devices."
        
        # Save bot response
        save_conversation(session['username'], response, 'bot')
        
        # Render template with response
        conversations = get_conversations(session['username'])
        return render_template('chatbot.html', 
                              conversations=conversations, 
                              intent=intent,
                              user_info=session)
    
    # Get existing conversations for display
    conversations = get_conversations(session['username'])
    return render_template('chatbot.html', 
                          conversations=conversations, 
                          intent=intent,
                          user_info=session)

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out!', 'info')
    return redirect(url_for('home'))

# Fix the intent responses to ensure the system always has responses
if not intent_responses:
    intent_responses = INTENT_RESPONSES

# Initialize the database on startup
init_db()

# Add a method to save and restore expanded dataset
def save_expanded_dataset(dataset):
    try:
        with open('expanded_customer_support_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print("Successfully saved expanded dataset")
        return True
    except Exception as e:
        print(f"Error saving expanded dataset: {e}")
        return False

# Admin route to manage dataset
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    global intent_responses
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Check if user is admin (simple check for demo)
    if session['username'] != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('chatbot'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_intent':
            intent_name = request.form.get('intent_name')
            response_text = request.form.get('response_text')
            
            # Add new intent response
            if intent_name and response_text:
                intent_responses[intent_name] = response_text
                save_expanded_dataset(intent_responses)
                flash(f'Added response for intent: {intent_name}', 'success')
        
        elif action == 'delete_intent':
            intent_name = request.form.get('delete_intent_name')
            
            # Remove intent response
            if intent_name in intent_responses:
                del intent_responses[intent_name]
                save_expanded_dataset(intent_responses)
                flash(f'Deleted response for intent: {intent_name}', 'success')
    
    return render_template('admin.html', intent_responses=intent_responses)

# API endpoint for updates
@app.route('/api/update', methods=['POST'])
def api_update():
    if request.is_json:
        data = request.get_json()
        if 'intent' in data and 'response' in data:
            # Update intent response
            intent_responses[data['intent']] = data['response']
            save_expanded_dataset(intent_responses)
            return {"status": "success", "message": "Intent response updated"}
    
    return {"status": "error", "message": "Invalid request format"}, 400

# Default chatbot responses for new users
@app.route('/api/default_response', methods=['GET'])
def default_response():
    return {"response": "Welcome to our customer support! How can I assist you today?"}

# Error handling for 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Error handling for 500 errors
@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Set to development mode
    app.run(debug=True)