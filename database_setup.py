import sqlite3

def setup_database():
    conn = sqlite3.connect('ecommerce.db')
    cursor = conn.cursor()

    # Existing tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            sold BOOLEAN NOT NULL DEFAULT 0,
            available_pieces INTEGER NOT NULL,
            image_url TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            item_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            total_price REAL NOT NULL,
            FOREIGN KEY (email) REFERENCES users (email)
        )
    ''')

    # Add the new "trends" table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_url TEXT NOT NULL,          -- URL of the video
            item_name TEXT NOT NULL,          -- Name of the trending item
            item_link TEXT NOT NULL,          -- URL to the product page
            thumbnail_url TEXT                -- URL of the video thumbnail
        )
    ''')

    # Insert initial items if needed
    initial_items = [
        ('T-shirt', 500, 0, 20, '/static/images/tshirt.jpg'),
        ('Watch', 1500, 0, 15, '/static/images/watch.jpg'),
        ('Mobile', 15000, 0, 10, '/static/images/mobile.jpg'),
        ('Shoe', 1200, 0, 25, '/static/images/shoe.jpg')
    ]

    # Check if the items table already has data
    cursor.execute('SELECT COUNT(*) FROM items')
    if cursor.fetchone()[0] == 0:
        cursor.executemany('''
            INSERT INTO items (name, price, sold, available_pieces, image_url)
            VALUES (?, ?, ?, ?, ?)
        ''', initial_items)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()
