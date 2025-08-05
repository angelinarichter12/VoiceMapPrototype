import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Warning: Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.")
    print("You can create a .env file with:")
    print("SUPABASE_URL=your_supabase_url")
    print("SUPABASE_ANON_KEY=your_supabase_anon_key")

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Supabase client: {e}")
        supabase = None

def get_supabase_client():
    """Get the Supabase client instance."""
    return supabase

def is_supabase_available():
    """Check if Supabase is properly configured."""
    return supabase is not None 