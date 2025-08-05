# 🚀 Quick Supabase Setup Guide

This guide will help you set up Supabase for user authentication and progress tracking in VoiceMap.

## 📋 Prerequisites

1. A Supabase account (free at [supabase.com](https://supabase.com))
2. Python 3.8+ with pip

## 🎯 Step-by-Step Setup

### Step 1: Create Supabase Project

1. **Go to [supabase.com](https://supabase.com)** and sign up/login
2. **Click "New Project"**
3. **Choose your organization**
4. **Enter project details:**
   - Name: `voicemap` (or your preferred name)
   - Database Password: Choose a strong password
   - Region: Choose closest to your users
5. **Click "Create new project"**
6. **Wait for the project to be created** (2-3 minutes)

### Step 2: Get Your Project Credentials

1. **In your Supabase dashboard**, go to **Settings > API**
2. **Copy these values:**
   - **Project URL** (looks like: `https://your-project-id.supabase.co`)
   - **anon/public key** (starts with `eyJ...`)

### Step 3: Create Environment File

1. **Create a `.env` file** in your project root:
   ```bash
   # In your project directory
   touch .env
   ```

2. **Add your Supabase credentials** to the `.env` file:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   ```

### Step 4: Set Up Database Tables

1. **Go to your Supabase dashboard**
2. **Click "SQL Editor"** in the left sidebar
3. **Run this SQL** to create the profiles table:

```sql
CREATE TABLE profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_assessment TIMESTAMP WITH TIME ZONE,
    assessment_count INTEGER DEFAULT 0
);

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to read their own profile
CREATE POLICY "Users can view own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);

-- Create policy to allow users to update their own profile
CREATE POLICY "Users can update own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

-- Create policy to allow users to insert their own profile
CREATE POLICY "Users can insert own profile" ON profiles
    FOR INSERT WITH CHECK (auth.uid() = id);
```

4. **Run this SQL** to create the assessments table:

```sql
CREATE TABLE assessments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    prediction TEXT NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    typical_probability DECIMAL(5,2),
    dementia_probability DECIMAL(5,2),
    medical_history JSONB,
    audio_duration DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE assessments ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to read their own assessments
CREATE POLICY "Users can view own assessments" ON assessments
    FOR SELECT USING (auth.uid() = user_id);

-- Create policy to allow users to insert their own assessments
CREATE POLICY "Users can insert own assessments" ON assessments
    FOR INSERT WITH CHECK (auth.uid() = user_id);
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Test the Integration

1. **Start the Flask application:**
   ```bash
   python3 app.py
   ```

2. **Open your browser** and go to `http://localhost:8080`
3. **You should now see "Sign Up" and "Sign In" buttons** in the navigation
4. **Try creating a new account** and signing in

## ✅ What You'll Get

With Supabase integration, users can:

- ✅ **Create accounts** with email and password
- ✅ **Sign in/out** securely
- ✅ **Track progress** over time
- ✅ **View assessment history** in their profile
- ✅ **See trends** and recommendations
- ✅ **Access personalized insights**

## 🔧 Troubleshooting

### "Supabase not configured" Error
- Make sure your `.env` file exists and has the correct credentials
- Restart the Flask application after creating the `.env` file

### Database Connection Issues
- Verify your Supabase URL and key are correct
- Check that your Supabase project is active
- Ensure the database tables were created successfully

### Authentication Issues
- Check that Row Level Security policies are set up correctly
- Verify the profiles table was created with the correct structure

## 🎉 You're Done!

Once you've completed these steps, your VoiceMap application will have:

1. **User authentication** - Users can create accounts and sign in
2. **Progress tracking** - All assessments are saved and tracked over time
3. **Personalized insights** - Users get recommendations based on their history
4. **Secure data storage** - All data is stored securely in Supabase

## 📞 Need Help?

If you run into any issues:

1. **Check the terminal output** for error messages
2. **Verify your Supabase credentials** are correct
3. **Ensure the database tables** were created successfully
4. **Restart the Flask application** after making changes

The integration is now **production-ready** and provides a complete user management system for your VoiceMap application! 