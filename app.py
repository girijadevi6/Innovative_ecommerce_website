from flask import Flask, render_template, request, redirect, session,jsonify,url_for
import smtplib
import os
import cv2 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import torch
import random
from ultralytics import YOLO
import re
import sqlite3
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management



def get_db_connection():
    conn = sqlite3.connect('ecommerce.db')
    conn.row_factory = sqlite3.Row
    return conn


# Load and clean your data
def clean_text(text):
    if pd.isnull(text):  # Check for NaN or missing values
        return "[missing data]"
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', '', text)
    text = text.capitalize()
    return text

# Assume you have your CSV loaded into `df`
data = pd.read_csv("cuisine_updated.csv", skipinitialspace=True)
df = pd.DataFrame(data)
df["ingredients"] = df["ingredients"].apply(clean_text)


@app.route('/')
def home():
    selected_category = request.args.get('category')  # Get the selected category from the query string
    conn = get_db_connection()

    # Fetch all items from the database
    query = """
    SELECT id, name, price, available_pieces, image_url, category
    FROM items
    """
    items = conn.execute(query).fetchall()

    # Fetch trending videos from the database

    trends_query = """
    SELECT video_url, item_name, item_link, thumbnail_url
    FROM trends limit 3
    """
    trends = conn.execute(trends_query).fetchall()

    conn.close()

    # Organize items by category
    categories = {}
    for item in items:
        if item['category'] not in categories:
            categories[item['category']] = []
        categories[item['category']].append(item)

    # Filter items for the selected category
    if selected_category:
        filtered_items = categories.get(selected_category, [])
    else:
        filtered_items = items  # Show all items if no category is selected

    # Cuisine, diet, and course dropdown options
    cuisines = df['cuisine'].unique().tolist()
    diets = df['diet'].unique().tolist()
    courses = ['Breakfast', 'Lunch', 'Dinner', 'Snack']  # Update this with more courses if needed

    return render_template(
        'index.html',
        categories=categories,
        items=filtered_items,
        selected_category=selected_category,
        logged_in='email' in session,
        cuisines=cuisines,
        diets=diets,
        courses=courses,
        trends=trends  # Pass the trends data
    )



def is_english(text):
    return bool(re.match(r'^[a-zA-Z0-9\s,.-]+$', text))

df = pd.DataFrame(data)

# Function to filter out non-English meals
df = df[df['name'].apply(is_english)]
df = df[df['description'].apply(is_english)]

@app.route('/select_cuisine', methods=['GET'])
def select_cuisine():
    # Fetch unique cuisines from the dataset
    cuisines = df['cuisine'].unique().tolist()
    return render_template('select_cuisine.html', cuisines=cuisines)

@app.route('/select_courses', methods=['POST'])
def select_courses():
    user_cuisine = request.form['cuisine']
    filtered_courses = df[df['cuisine'].str.lower() == user_cuisine.lower()]['course'].unique()
    return render_template('select_courses.html', courses=filtered_courses, cuisine=user_cuisine)



@app.route('/generate_meal_plan', methods=['POST'])
def generate_meal_plan():
    user_cuisine = request.form['cuisine']
    user_courses = request.form.getlist('course')  # Get the selected courses

    filtered_df = df[df['cuisine'].str.lower() == user_cuisine.lower()]

    if filtered_df.empty:
        return "No meals match your preferences. Please try different inputs."

    conn = get_db_connection()
    items = conn.execute('SELECT id, name FROM items').fetchall()
    conn.close()

    # Extract item names and their IDs (normalized to lowercase)
    item_mapping = {item['name'].strip().lower(): item['id'] for item in items}

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    meal_plan = []

    for day in days:
        for course in user_courses:
            meal_options = filtered_df[filtered_df['course'].str.lower() == course.lower()]
            if not meal_options.empty:
                selected_meal = meal_options.sample(1).iloc[0]

                # Process ingredients to create links for available items
                ingredient_links = []
                for ingredient in selected_meal['ingredients'].split(', '):  # Assuming ingredients are comma-separated
                    words = ingredient.split()  # Split ingredient into individual words
                    ingredient_link_parts = []

                    for word in words:
                        # Remove any punctuation or extra spaces from each word
                        word = word.strip().lower()

                        # Check if the word exists in the item mapping
                        ingredient_id = item_mapping.get(word)

                        if ingredient_id:
                            # Create a clickable link for the matching item
                            ingredient_link_parts.append(f"<a href='/order/{ingredient_id}'>{word}</a>")
                        else:
                            # If no match, just display the word as is
                            ingredient_link_parts.append(word)

                    # Join the parts of the ingredient back together
                    ingredient_links.append(" ".join(ingredient_link_parts))

                meal_plan.append({
                    'Day': day,
                    'Course': course,
                    'Meal Name': selected_meal['name'],
                    'Description': selected_meal['description'],
                    'Prep Time (mins)': selected_meal['prep_time'],
                    'Ingredients': ', '.join(ingredient_links),  # Join links into a single string
                    'Instructions': selected_meal['instructions'],
                    'Image URL': selected_meal['image_url']
                })

    return render_template('meal_plan.html', meal_plan=meal_plan)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")

# Load the recipes dataset
df = pd.read_csv("cuisine_updated.csv")
df["ingredient_list"] = df["ingredients"].fillna("").str.lower().str.split(", ")

@app.route("/upload")
def upload_page():
    """Route to display the upload page."""
    return render_template("upload.html")

@app.route("/detect_recipe", methods=["POST"])
def detect_recipe():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    image_files = request.files.getlist("images")

    if not image_files or all(image.filename == "" for image in image_files):
        return jsonify({"error": "No selected files"}), 400

    detected_ingredients = set()

    for image_file in image_files:
        image_path = os.path.join("static", image_file.filename)
        image_file.save(image_path)

        # Detect ingredients in the uploaded images
        detected_ingredients.update(detect_ingredients(image_path))

    # Get matching recipes
    recipes = find_best_matching_recipes(detected_ingredients)

    return render_template("recipe_results.html", ingredients=list(detected_ingredients), recipes=recipes)

def detect_ingredients(image_path):
    """Detects ingredients in an image using YOLOv8 and returns a list of detected ingredient names."""
    results = model(image_path, device=device, verbose=False)  # Run YOLO inference

    detected_ingredients = set()
    for box in results[0].boxes:
        class_id = int(box.cls)
        detected_ingredients.add(model.names[class_id].lower())

    return list(detected_ingredients)

def find_best_matching_recipes(detected_ingredients):
    """Find recipes that contain the most detected ingredients with clickable links."""
    df["match_count"] = df["ingredient_list"].apply(lambda x: sum(ing in x for ing in detected_ingredients))
    sorted_recipes = df[df["match_count"] > 0].sort_values(by="match_count", ascending=False)

    conn = get_db_connection()
    items = conn.execute('SELECT id, name FROM items').fetchall()
    conn.close()

    # Create a mapping of item names to their IDs (normalized to lowercase)
    item_mapping = {item['name'].strip().lower(): item['id'] for item in items}

    formatted_recipes = []
    for _, row in sorted_recipes.iterrows():
        ingredient_links = []
        for ingredient in row["ingredients"].split(', '):  # Assuming ingredients are comma-separated
            words = ingredient.split()
            ingredient_link_parts = []

            for word in words:
                word_cleaned = word.strip().lower()
                ingredient_id = item_mapping.get(word_cleaned)

                if ingredient_id:
                    ingredient_link_parts.append(f"<a href='/order/{ingredient_id}' style='text-decoration: none; color: inherit;'>{word}</a>")
                else:
                    ingredient_link_parts.append(word)

            ingredient_links.append(" ".join(ingredient_link_parts))

        formatted_recipes.append({
            "name": row["name"],
            "instructions": row["instructions"],
            "ingredients": ", ".join(ingredient_links),  # Links properly formatted
            "prep_time": row["prep_time"]
        })

    return formatted_recipes

@app.route('/recipe_results')
def recipe_results():
    detected_ingredients = ["salt", "tomato", "onion"]  # Example detected ingredients
    recipes = find_best_matching_recipes(detected_ingredients)
    return render_template('recipe_results.html', recipes=recipes)


UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Upload video page
@app.route('/upload_vi', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # Get form data
        item_name = request.form['item_name']
        item_link = request.form['item_link']
        
        # Save video file
        video = request.files['video']
        thumbnail = request.files['thumbnail']
        email=request.form['email']

        if video and thumbnail:
            video_filename = secure_filename(video.filename)
            thumbnail_filename = secure_filename(thumbnail.filename)
            
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], thumbnail_filename)
            
            video.save(video_path)
            thumbnail.save(thumbnail_path)
            
            # Store data in database
            conn = get_db_connection()
            conn.execute('INSERT INTO trends (video_url, item_name, item_link, thumbnail_url,user_email) VALUES (?, ?, ?, ?,?)',
                         (video_path, item_name, item_link, thumbnail_path,email))
            conn.commit()
            conn.close()
            
            return redirect(url_for('trends'))
    
    return render_template('upload_vi.html')

# Trends page (Displays all videos)
@app.route('/trends')
def trends():
    conn = get_db_connection()

    # Fetch all trend items from the trends table
    query = """
    SELECT video_url, item_name, item_link
    FROM trends limit 3
    """
    trends_data = conn.execute(query).fetchall()
    conn.close()

    # Prepare trends data for rendering
    trends = []
    for trend in trends_data:
        trends.append((trend['video_url'], trend['item_name'], trend['item_link']))

    return render_template('trends.html', trends=trends)


def update_views(video_url):
    conn = sqlite3.connect('ecommerce.db')
    cursor = conn.cursor()

    # Get video details
    cursor.execute("SELECT views, user_email FROM trends WHERE video_url=?", (video_url,))
    result = cursor.fetchone()

    if result:
        views, user_email = result
        views += 1  # Increment view count

        # Update views in the database
        cursor.execute("UPDATE trends SET views=? WHERE video_url=?", (views, video_url))

        # Add coins (1 coin per 10 views)
        if views % 10 == 0:  
            cursor.execute("UPDATE users SET coins = coins + 1 WHERE email=?", (user_email,))

        conn.commit()
    
    conn.close()


@app.route('/update_views', methods=['POST'])
def update_video_views():
    data = request.json
    video_url = data.get('video_url')

    if video_url:
        update_views(video_url)
        return jsonify({"message": "View updated successfully"}), 200
    return jsonify({"error": "Invalid request"}), 400

def get_user_coins(user_email):
    conn = sqlite3.connect("ecommerce.db")
    cursor = conn.cursor()
    cursor.execute("SELECT coins FROM users WHERE email = ?", (user_email,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0  # Return coins or 0 if no record exists

@app.route('/coins_earned')
def coins_earned():
    if 'email' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in

    user_email = session['email']
    coins = get_coins_earned(user_email)

    return render_template("coins_earned.html", coins=coins)


@app.route('/shorts_page')
def shorts_page():
    conn = get_db_connection()

    # Fetch all trend items from the trends table
    query = """
    SELECT video_url, item_name, item_link, thumbnail_url
    FROM trends
    """
    trends_data = conn.execute(query).fetchall()
    conn.close()

    # Prepare trends data for rendering
    shorts = []
    for trend in trends_data:
        shorts.append({
            'video_url': trend['video_url'],
            'item_name': trend['item_name'],
            'item_link': trend['item_link'],
            'thumbnail_url': trend['thumbnail_url']
        })

    return render_template('shorts.html', trends=shorts)



@app.route('/category/<category_name>')
def category_items(category_name):
    print(f"Category route: Selected category: {category_name}")  # Debug log
    items = get_items_by_category(category_name)
    categories = ["Men's Fashion", "Women's Fashion", "Grocery", "Beauty", "Sports"]
    return render_template(
        'index.html',
        categories=categories,
        items=items,
        selected_category=category_name,
        logged_in=session.get('email') is not None
    )

def get_items_by_category(category_name):
    print(f"Fetching items for category: {category_name}")  # Debug log
    conn = get_db_connection()
    query = "SELECT id, name, price, available_pieces, image_url, category FROM items WHERE category = ?"
    items = conn.execute(query, (category_name,)).fetchall()
    conn.close()
    print(f"Fetched items: {items}")  # Debug log
    return [dict(item) for item in items]  # Convert to dictionary



@app.route("/index")
def index():
    user_email = session.get("user_email")
    category = request.args.get('category', "Men's Fashion")  # Default category
    print(f"Index route: Selected category: {category}")  # Debug log

    items = get_items_by_category(category)
    categories = ["Men's Fashion", "Women's Fashion", "Grocery", "Beauty", "Sports"]
    top_trends = trends[:3] if trends else []

    return render_template(
        'index.html',user_email=user_email,
        categories=categories,
        items=items,
        selected_category=category,
        trends=top_trends,  # Pass only the top 3 trends
        logged_in=session.get('email') is not None
    )

@app.route('/fashion')
def fashion():
    return render_template('fashion.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        conn.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
        conn.commit()
        conn.close()
        return redirect('/login')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password)).fetchone()
        conn.close()

        if user:
            session['email'] = email
            return redirect('/')
        else:
            return 'Invalid credentials'

    return render_template('login.html')

def get_coins_earned(user_email):
    conn = sqlite3.connect("ecommerce.db")  # Replace with your actual database file
    cursor = conn.cursor()
    
    cursor.execute("SELECT coins FROM users WHERE email = ?", (user_email,))
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else 0  # Return coins if found, else return 0

@app.route('/watch/<int:video_id>')
def watch_video(video_id):
    update_views_and_coins(video_id)  # Increment views and coins
    return "Video watched, coins updated!"  # You can redirect to actual video page

def update_views_and_coins(video_id):
    conn = sqlite3.connect("ecommerce.db")
    cursor = conn.cursor()

    # Get current views and uploader's email
    cursor.execute("SELECT views, user_email FROM trends WHERE id = ?", (video_id,))
    result = cursor.fetchone()

    if result:
        current_views, user_email = result
        new_views = current_views + 1

        # Update views count in `trends` table
        cursor.execute("UPDATE trends SET views = ? WHERE id = ?", (new_views, video_id))

        # If views is a multiple of 10, increment coins in `users` table
        if new_views % 10 == 0:
            cursor.execute("UPDATE users SET coins = coins + 1 WHERE email = ?", (user_email,))

    conn.commit()
    conn.close()



@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/')


@app.route('/add', methods=['GET', 'POST'])
def add_item():
    if 'email' not in session or session['email'] != 'admin@gmail.com':
        return redirect('/login')

    if request.method == 'POST':
        name = request.form['name']
        price = request.form['price']
        available_pieces = request.form['available_pieces']
        image_url = request.form['image_url']
        category = request.form['category']

        conn = get_db_connection()
        conn.execute(
            'INSERT INTO items (name, price, available_pieces, image_url, category) VALUES (?, ?, ?, ?, ?)',
            (name, price, available_pieces, image_url, category)
        )
        conn.commit()
        conn.close()

        return redirect('/')
    
    return render_template('add_item.html')


@app.route('/update/<int:item_id>', methods=['GET', 'POST'])
def update_item(item_id):
    if 'email' not in session or session['email'] != 'admin@gmail.com':
        return redirect('/login')  # Only admin can update items

    conn = get_db_connection()
    item = conn.execute('SELECT * FROM items WHERE id = ?', (item_id,)).fetchone()

    if request.method == 'POST':
        name = request.form['name']
        price = request.form['price']
        available_pieces = request.form['available_pieces']
        image_url = request.form['image_url']
        category=request.form['category']

        conn.execute('''
            UPDATE items 
            SET name = ?, price = ?, available_pieces = ?, image_url = ? ,category=?
            WHERE id = ?
        ''', (name, price, available_pieces, image_url,category, item_id))
        conn.commit()
        conn.close()
        return redirect('/')

    conn.close()
    return render_template('update_item.html', item=item)


@app.route('/delete/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    if 'email' not in session or session['email'] != 'admin@gmail.com':
        return redirect('/login')  # Only admin can delete items

    conn = get_db_connection()
    conn.execute('DELETE FROM items WHERE id = ?', (item_id,))
    conn.commit()
    conn.close()
    return redirect('/')





def send_order_email(user_email, order_details):
    # SMTP server details (example: Gmail)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587  # For TLS
    sender_email = "dgirija651@gmail.com"  # Your email address
    sender_password = "hslq iqzd keko jyxb"  # Your email password (or app password if using Gmail)
    
    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = user_email
    msg['Subject'] = "Your Order Confirmation"

    # Order details to send in the email body
    order_info = f"Hello,\n\nThank you for your order! Here are your order details:\n\n{order_details}\n\nBest Regards,\nYour Company"
    
    msg.attach(MIMEText(order_info, 'plain'))
    
    try:
        # Connect to the Gmail SMTP server and send the email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, user_email, text)
        server.quit()  # Terminate the session
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


@app.route('/order/<int:item_id>', methods=['GET', 'POST'])
def place_order(item_id):
    if 'email' not in session:
        return redirect('/login')  # Ensure the user is logged in before ordering

    conn = get_db_connection()
    item = conn.execute('SELECT * FROM items WHERE id = ?', (item_id,)).fetchone()

    if not item:
        conn.close()
        return "Item not found."

    if request.method == 'POST':
        email = session['email']
        quantity = int(request.form['quantity'])

        if item['available_pieces'] < quantity:
            conn.close()
            return "Not enough items in stock."

        total_price = item['price'] * quantity

        # Insert the order into the orders table
        conn.execute(''' 
            INSERT INTO orders (email, item_name, quantity, total_price)
            VALUES (?, ?, ?, ?)
        ''', (email, item['name'], quantity, total_price))

        # Update the available pieces in the items table
        conn.execute('''
            UPDATE items SET available_pieces = available_pieces - ? WHERE id = ?
        ''', (quantity, item_id))

        conn.commit()

        # Prepare order details to send in email
        order_details = f"""
        Order Number: #12345 (or generate dynamically)
        Product: {item['name']}
        Quantity: {quantity}
        Total Price: ${total_price}
        """

        # Send email to the user with order details
        try:
            send_order_email(email, order_details)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")

        conn.close()
        return redirect('/')

    conn.close()
    return render_template('order_form.html', item=item)


@app.route('/my_orders')
def my_orders():
    if 'email' not in session:
        return redirect('/login') # Ensure the user is logged in

    email = session['email']

    conn = get_db_connection()
    orders = conn.execute('SELECT * FROM orders WHERE email = ?', (email,)).fetchall()
    conn.close()

    total_price = sum(order['total_price'] for order in orders)
    
    return render_template('my_orders.html', orders=orders, total_price=total_price)



if __name__ == '__main__':
    app.run(debug=True)
