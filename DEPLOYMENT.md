# VoiceMap Deployment Guide

This guide will help you deploy your VoiceMap application to make it publicly accessible on the internet.

## Option 1: Deploy to Render (Recommended - Free)

### Step 1: Prepare Your Repository
1. Make sure your GitHub repository is public
2. Ensure all files are committed and pushed to GitHub

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" and select "Web Service"
3. Connect your GitHub account and select your VoiceMap repository
4. Configure the deployment:
   - **Name**: `voicemap` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free

5. Click "Create Web Service"
6. Wait for the build to complete (usually 5-10 minutes)

### Step 3: Access Your Website
- Your website will be available at: `https://your-app-name.onrender.com`
- Share this URL with anyone to access VoiceMap

## Option 2: Deploy to Heroku (Alternative)

### Step 1: Install Heroku CLI
```bash
# macOS
brew install heroku/brew/heroku

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### Step 2: Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-voicemap-app

# Add Python buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open the app
heroku open
```

## Option 3: Deploy to Railway (Alternative - Free)

1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway will automatically detect it's a Python app
4. Deploy and get a public URL

## Important Notes

### Model Files
- Make sure your trained model files (`models/cnn_model.pt`) are included in the repository
- The deployment will need access to these files

### Environment Variables
- For production, consider setting environment variables for any sensitive data
- Render/Heroku will automatically handle the PORT environment variable

### Performance
- The free tier of Render may have some limitations
- Audio processing can be resource-intensive
- Consider upgrading to a paid plan for better performance

### Troubleshooting
- Check the deployment logs if the app doesn't start
- Ensure all dependencies are in `requirements.txt`
- Make sure the model files are accessible

## Testing Your Deployment

1. Visit your public URL
2. Try recording audio samples
3. Verify the results match your local testing
4. Test with different browsers/devices

## Security Considerations

- The app is designed for research/demo purposes
- Consider adding authentication if needed
- Be aware that audio data is processed on the server
- Consider data privacy implications

## Support

If you encounter issues:
1. Check the deployment logs
2. Verify all files are properly committed
3. Ensure the model files are included
4. Test locally first to ensure everything works 